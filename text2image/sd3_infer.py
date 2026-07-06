import math
import os
import numpy as np
import sd3_impls
import torch
from PIL import Image
from safetensors import safe_open
from sd3_impls import (
    SDVAE,
    BaseModel,
    Denoiser,
    SD3LatentFormat,
)

#################################################################################################
### Wrappers for model parts
#################################################################################################


def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in ckpt.keys():
        model_key = key
        if remap is not None and key in remap:
            model_key = remap[key]
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None:
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device)
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                # print(f"K: {model_key}, O: {obj.shape} T: {tensor.shape}")
                if obj.shape != tensor.shape:
                    print(
                        f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}"
                    )
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e

class SD3:
    def __init__(
        self, model, shift, verbose=False, device="cpu"
    ):

        with safe_open(model, framework="pt", device="cpu") as f:
            self.model = BaseModel(
                shift=shift,
                file=f,
                prefix="model.diffusion_model.",
                device="cuda",
                dtype=torch.float16,
                verbose=verbose,
            ).eval()
            load_into(f, self.model, "model.", "cuda", torch.float16)

class VAE:
    def __init__(self, model, dtype: torch.dtype = torch.float16):
        with safe_open(model, framework="pt", device="cpu") as f:
            self.model = SDVAE(device="cpu", dtype=dtype).eval().cpu()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, "cpu", dtype)


#################################################################################################
### Main inference logic
#################################################################################################

# Note: Sigma shift value, publicly released models use 3.0
SHIFT = 3.0
# Naturally, adjust to the width/height of the model you have
WIDTH = 512
HEIGHT = 512
# Pick your prompt
PROMPT = "a photo of a cat"
STEPS = 28
# Seed
SEED = 23
# SEEDTYPE = "fixed"
SEEDTYPE = "rand"
# SEEDTYPE = "roll"
# Actual model file path
MODEL = "models/sd3.5_medium.safetensors"
# VAE model file path, or set None to use the same model file
VAEFile = None  # "models/sd3_vae.safetensors"
# Optional init image file path
INIT_IMAGE = None
# If init_image is given, this is the percentage of denoising steps to run (1.0 = full denoise, 0.0 = no denoise at all)
DENOISE = 0.8
# Output file path
OUTDIR = "outputs"
# SAMPLER
SAMPLER = "dpmpp_2m"
# MODEL FOLDER
MODEL_FOLDER = "models"

class SD3Inferencer:

    def __init__(self):
        self.verbose = False

    def print(self, txt):
        if self.verbose:
            print(txt)

    def load(
        self,
        model=MODEL,
        vae=VAEFile,
        shift=SHIFT,
        verbose=False,
    ):
        self.verbose = verbose
        print(f"Loading SD3 model {os.path.basename(model)}...")
        self.sd3 = SD3(model, shift, device="cuda")
        print("Loading VAE model...")
        self.vae = VAE(vae or model)
        print("Loading conditioning...")
        self.conditioning = torch.load("conditioning.pt", map_location="cpu")
        print("Models loaded.")

    def get_empty_latent(self, batch_size, width, height, seed, device="cuda"):
        self.print("Prep an empty latent...")
        shape = (batch_size, 16, height // 8, width // 8)
        latents = torch.zeros(shape, device=device)
        for i in range(shape[0]):
            prng = torch.Generator(device=device).manual_seed(int(seed + i))
            latents[i] = torch.randn(shape[1:], generator=prng, device=device)
        return latents

    def get_sigmas(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    def get_noise(self, seed, latent):
        generator = torch.manual_seed(seed)
        self.print(
            f"dtype = {latent.dtype}, layout = {latent.layout}, device = {latent.device}"
        )
        return torch.randn(
            latent.size(),
            dtype=torch.float32,
            layout=latent.layout,
            generator=generator,
            device="cpu",
        ).to(latent.dtype)

    def max_denoise(self, sigmas):
        max_sigma = float(self.sd3.model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def fix_cond(self, cond):
        cond, pooled = (cond[0].half().cuda(), cond[1].half().cuda())
        return {"c_crossattn": cond, "y": pooled}

    def do_sampling(
        self,
        latent,
        seed,
        conditioning,
        steps,
        sampler="dpmpp_2m",
        denoise=1.0,
    ) -> torch.Tensor:
        self.print("Sampling...")
        latent = latent.half().cuda()
        self.sd3.model = self.sd3.model.cuda()
        noise = self.get_noise(seed, latent).cuda()
        sigmas = self.get_sigmas(self.sd3.model.model_sampling, steps).cuda()
        sigmas = sigmas[int(steps * (1 - denoise)) :]
        conditioning = self.fix_cond(conditioning)
        noise_scaled = self.sd3.model.model_sampling.noise_scaling(
            sigmas[0], noise, latent, self.max_denoise(sigmas)
        )
        sample_fn = getattr(sd3_impls, f"sample_{sampler}")
        denoiser = Denoiser
        latent = sample_fn(
            denoiser(self.sd3.model, steps),
            noise_scaled,
            sigmas,
            conditioning,
        )
        latent = SD3LatentFormat().process_out(latent)
        self.sd3.model = self.sd3.model.cpu()
        self.print("Sampling done")
        return latent

    def vae_encode(
        self, image
    ) -> torch.Tensor:
        self.print("Encoding image to latent...")
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        image_torch = torch.from_numpy(batch_images).cuda()
        image_torch = 2.0 * image_torch - 1.0
        image_torch = image_torch.cuda()
        self.vae.model = self.vae.model.cuda()
        latent = self.vae.model.encode(image_torch).cpu()
        self.vae.model = self.vae.model.cpu()
        self.print("Encoded")
        return latent

    def vae_encode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.unsqueeze(0)
        latent = SD3LatentFormat().process_in(latent)
        return latent

    def vae_decode(self, latent) -> Image.Image:
        self.print("Decoding latent to image...")
        latent = latent.cuda()
        self.vae.model = self.vae.model.cuda()
        image = self.vae.model.decode(latent)
        image = image.float()
        self.vae.model = self.vae.model.cpu()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        out_image = Image.fromarray(decoded_np)
        self.print("Decoded")
        return out_image

    def _image_to_latent(
        self,
        image,
        width,
        height,
    ) -> torch.Tensor:
        image_data = Image.open(image)
        image_data = image_data.resize((width, height), Image.LANCZOS)
        latent = self.vae_encode(image_data)
        latent = SD3LatentFormat().process_in(latent)
        return latent

    def gen_image(
        self,
        width=WIDTH,
        height=HEIGHT,
        steps=STEPS,
        sampler=SAMPLER,
        seed=SEED,
        seed_type=SEEDTYPE,
        out_dir=OUTDIR,
        init_image=INIT_IMAGE,
        denoise=DENOISE,
    ):
        if init_image:
            latent = self._image_to_latent(init_image, width, height)
        else:
            latent = self.get_empty_latent(1, width, height, seed, "cpu")
            latent = latent.cuda()
        seed_num = None

        if seed_type == "roll":
            seed_num = seed if seed_num is None else seed_num + 1
        elif seed_type == "rand":
            seed_num = torch.randint(0, 100000, (1,)).item()
        else:  # fixed
            seed_num = seed
        
        sampled_latent = self.do_sampling(
            latent,
            seed_num,
            self.conditioning,
            steps,
            sampler,
            denoise if init_image else 1.0,
        )
        image = self.vae_decode(sampled_latent)
        save_path = os.path.join(out_dir, f"0.png")
        self.print(f"Saving to to {save_path}")
        image.save(save_path)
        self.print("Done")

CONFIGS = {
    "sd3.5_medium": {
        "shift": 3.0,
        "steps": 28,
        "sampler": "dpmpp_2m",
    },
}

@torch.no_grad()
def main(
    model=MODEL,
    out_dir=OUTDIR,
    seed=SEED,
    seed_type=SEEDTYPE,
    sampler=None,
    steps=None,
    shift=None,
    width=WIDTH,
    height=HEIGHT,
    vae=VAEFile,
    init_image=INIT_IMAGE,
    denoise=DENOISE,
    verbose=False,
    **kwargs,
):
    assert not kwargs, f"Unknown arguments: {kwargs}"

    config = CONFIGS.get(os.path.splitext(os.path.basename(model))[0], {})
    _shift = shift or config.get("shift", 3)
    _steps = steps or config.get("steps", 50)
    _sampler = sampler or config.get("sampler", "dpmpp_2m")

    inferencer = SD3Inferencer()

    inferencer.load(
        model,
        vae,
        _shift,
        verbose,
    )

    out_dir = 'generated_images'

    os.makedirs(out_dir, exist_ok=True)

    inferencer.gen_image(
        width,
        height,
        _steps,
        _sampler,
        seed,
        seed_type,
        out_dir,
        init_image,
        denoise,
    )


if __name__ == "__main__":
    main()
