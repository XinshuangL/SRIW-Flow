#!/usr/bin/env python3

import argparse
import numpy as np
import torch
from torch.cuda.amp import autocast
from sd3_impls import BaseModel, ModelSamplingDiscreteFlow, SD3LatentFormat, SDVAE
from safetensors import safe_open
from sd3_infer import load_into
import os
from PIL import Image


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fix_cond(cond, B: int):
    return {"c_crossattn": cond[0].repeat(B, 1, 1), "y": cond[1].repeat(B, 1)}


@torch.no_grad()
def generate_image(
    diffusion_model,
    vae,
    latent_fmt,
    sampling,
    steps: int = 28,
    device: torch.device = torch.device("cuda"),
    cond=None,
    seed: int = 1234,
    B: int = 1,
    image_L: int = 512,
):
    torch.manual_seed(seed)

    h, w = image_L // 8, image_L // 8
    x = torch.randn((B, 16, h, w), device=device, dtype=torch.float16)

    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float16)

    for i in range(steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        t = sampling.timestep(sigma.view(1))

        with autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                v = diffusion_model(
                    x,
                    t.to(device=device, dtype=torch.float32),
                    context=cond["c_crossattn"].to(device=x.device, dtype=x.dtype),
                    y=cond["y"].to(device=x.device, dtype=x.dtype),
                    skip_layers=[],
                ).float()

        d_sigma = sigma_next - sigma
        x = x + d_sigma * v

    with autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
        lat_out = latent_fmt.process_out(x)
        img_out = vae.decode(lat_out)

    return img_out, x.detach().cpu()


def build_argparser():
    p = argparse.ArgumentParser("SD3.5-Medium IID text-to-image generation (flow-matching)")
    p.add_argument("--model", type=str, default="models/sd3.5_medium.safetensors", help="Path to sd3.5 .safetensors (contains diffusion + VAE)")
    p.add_argument("--image_L", type=int, default=512)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--B", type=int, default=10)
    return p

def main():
    test_base_seed = 10000000

    args = build_argparser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("[Load] SD3.5 base:", args.model)
    with safe_open(args.model, framework="pt", device="cpu") as f:
        base = BaseModel(
            shift=3.0,
            file=f,
            prefix="model.diffusion_model.",
            dtype=torch.float16,
            verbose=False,
        )
        load_into(f, base, "model.", "cpu", torch.float16)

    print("[Load] VAE (from same file)")
    with safe_open(args.model, framework="pt", device="cpu") as f:
        vae = SDVAE(dtype=torch.float16, device="cpu").eval().cpu()
        prefix = ""
        if any(k.startswith("first_stage_model.") for k in f.keys()):
            prefix = "first_stage_model."
        load_into(f, vae, prefix, "cpu", torch.float16)

    base = base.to(device)
    base.eval()
    vae = vae.to(device).eval()

    latent_fmt = SD3LatentFormat()
    sampling = ModelSamplingDiscreteFlow(shift=3.0)

    cond_str_list = [
        "a releastic cat",
        "a cat",
        "a fish",
        "a realistic fish",
        "something outside",
        "someone ahead",
    ]
    for cond_str in cond_str_list:
        print("[Load] Conditioning:", cond_str)
        cond_raw = torch.load(f"conditionings/{cond_str}.pt", map_location="cpu")
        cond = fix_cond(cond_raw, args.B)

        for repeat_id in range(1000):
            cur_seed = test_base_seed + repeat_id

            os.makedirs(f"iid/{cond_str}", exist_ok=True)
            seed_everything(cur_seed)
            out_t, samples = generate_image(
                diffusion_model=base.diffusion_model,
                vae=vae,
                latent_fmt=latent_fmt,
                sampling=sampling,
                image_L=args.image_L,
                steps=args.steps,
                device=device,
                cond=cond,
                seed=cur_seed,
                B=args.B,
            )
            out_t = out_t[0]
            image = torch.clamp((out_t + 1.0) / 2.0, min=0.0, max=1.0)
            decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
            decoded_np = decoded_np.astype(np.uint8)
            out_image = Image.fromarray(decoded_np)
            out_image.save(f"iid/{cond_str}/{repeat_id}.png")
            torch.save(samples, f"iid/{cond_str}/samples_{repeat_id}.pt")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
