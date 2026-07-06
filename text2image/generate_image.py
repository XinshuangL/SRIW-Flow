#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import torch
from torch.cuda.amp import autocast
from sd3_impls import BaseModel, ModelSamplingDiscreteFlow, SD3LatentFormat, SDVAE
from safetensors import safe_open
from sd3_infer import load_into
import os
from PIL import Image
import math
import torch.nn.functional as F
from tqdm import tqdm


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fix_cond(cond, B: int):
    return {"c_crossattn": cond[0].repeat(B, 1, 1), "y": cond[1].repeat(B, 1)}


def normalize_K(K: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    upper_tri_mask = torch.triu(torch.ones_like(K, dtype=torch.bool), diagonal=1)
    K_upper = K[upper_tri_mask]
    median_val = torch.median(K_upper).detach() + eps
    return K / median_val, K_upper / median_val

def compute_K(x: torch.Tensor) -> torch.Tensor:
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    K = x_norm + x_norm.T - 2 * x @ x.T
    return normalize_K(K)

def logdet_psd_cholesky(K: torch.Tensor, jitter: float = 1e-3) -> torch.Tensor:
    B = K.size(0)
    old_dtype = K.dtype
    K = K.to(dtype=torch.float32)
    I = torch.eye(B, dtype=K.dtype, device=K.device)
    L, _ = torch.linalg.cholesky_ex(K + jitter * I)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
    return logdet.to(dtype=old_dtype)

def compute_diversity_and_force(
    diversity_name: str,
    latents: torch.Tensor,
    sigma: float,
    abs_dt: float,
    *,
    velocity: torch.Tensor = None,
    f_0: float = 1.0,
    s_proj: str = "soft",
):
    old_dtype = latents.dtype
    latents = latents.to(torch.float32)
    velocity = velocity.to(torch.float32)

    B, C, H, W = latents.shape
    assert B >= 2, "B must be greater than or equal to 2"

    score = (velocity * (1 - sigma) - latents) / sigma

    latent_vector = latents.mean(dim=[2, 3])
    velocity_vector = velocity.mean(dim=[2, 3])
    score_vector = score.mean(dim=[2, 3])

    with torch.enable_grad():
        l_req = latent_vector.detach().requires_grad_(True)
        xt = l_req + sigma * velocity_vector

        if diversity_name == "dpp":
            jitter = 1e-3
            K, _ = compute_K(xt)
            L = torch.exp(-K)
            I = torch.eye(B, dtype=L.dtype, device=L.device)
            diversity = logdet_psd_cholesky(L, jitter=jitter) - logdet_psd_cholesky(L + I, jitter=jitter)
        else:
            raise ValueError(f"Unsupported diversity name: {diversity_name}")

        (grad_latents,) = torch.autograd.grad(diversity, l_req, create_graph=False)
        f = grad_latents
        f = f.detach()

    score_vector_unit = score_vector / (score_vector.norm(dim=1, keepdim=True) + 1e-8)

    if s_proj == 'hard':
        f_parallel_scalar = (f * score_vector_unit).sum(dim=1, keepdim=True)
        f_parallel = f_parallel_scalar * score_vector_unit
        f_vertical = f - f_parallel
        f = f_vertical + F.relu(f_parallel_scalar) * score_vector_unit
    elif s_proj == 'soft':
        f_parallel_scalar = (f * score_vector_unit).sum(dim=1, keepdim=True)
        f_parallel = f_parallel_scalar * score_vector_unit
        f_vertical = f - f_parallel
        soft_factor = sigma
        f_parallel_scalar_soft = f_parallel_scalar * soft_factor + F.relu(f_parallel_scalar) * (1.0 - soft_factor)
        f = f_vertical + f_parallel_scalar_soft * score_vector_unit
    elif s_proj == 'none':
        pass
    else:
        raise ValueError(f"Invalid s_proj: {s_proj}")

    f = f / (f.view(-1).norm() + 1e-8)
    f = f * (velocity_vector.view(-1).norm() + 1e-8)

    time_scale = math.sqrt(max(float(sigma), 0.0) + 1e-12)
    combined_force = f_0 * time_scale * f

    combined_force_NCHW = combined_force.view(B, C, 1, 1).repeat(1, 1, H, W)

    to_save = {
        'latent_vector': latent_vector.detach().cpu(),
        'velocity_vector': velocity_vector.detach().cpu(),
        'score_vector': score_vector.detach().cpu(),
        'combined_force': combined_force.detach().cpu(),
        'diversity': diversity.detach().cpu(),
        'sigma': sigma,
        'abs_dt': abs_dt,
    }

    return diversity.detach().to(old_dtype), combined_force_NCHW.detach().to(old_dtype), to_save

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
    diversity_type: str = "none",
    f_0: float = 1.0,
    s_proj: str = "soft",
):
    torch.manual_seed(seed)

    h, w = image_L // 8, image_L // 8
    x = torch.randn((B, 16, h, w), device=device, dtype=torch.float16)

    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float16)

    to_save_list = []
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

        if not diversity_type == "none":
            _, force, to_save = compute_diversity_and_force(
                diversity_name=diversity_type,
                latents=x,
                sigma=sigma,
                abs_dt=abs(d_sigma),
                velocity=-v,
                f_0=f_0,
                s_proj=s_proj,
            )
        else:
            force = 0.0
            to_save = None
        if to_save is not None:
            to_save_list.append(to_save)

        x = x + d_sigma * v + abs(d_sigma) * force

    with autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
        lat_out = latent_fmt.process_out(x)
        img_out = vae.decode(lat_out)

    return img_out, \
        {"to_save_list": to_save_list, 'samples': x.detach().cpu()}

def build_argparser():
    p = argparse.ArgumentParser(description="SD3.5-Medium diverse joint text-to-image sampling")
    p.add_argument("--model", type=str, default="models/sd3.5_medium.safetensors", help="Path to sd3.5 .safetensors (contains diffusion + VAE)")
    p.add_argument("--image_L", type=int, default=512)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--B", type=int, default=10)
    p.add_argument("--s_proj", choices=["none", "soft", "hard"], required=True,
                   help="score-regularization mode (SR): none, soft, or hard")
    return p

def main():
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

    f_0 = 0.1

    cond_str_list = [
        "a releastic cat",
        "a cat",
        "a fish",
        "a realistic fish",
        "something outside",
        "someone ahead",
    ]
    for repeat_id in tqdm(range(1000)):
        for cond_str in cond_str_list:
            print("[Load] Conditioning:", cond_str)
            cond_raw = torch.load(f"conditionings/{cond_str}.pt", map_location="cpu")
            cond = fix_cond(cond_raw, args.B)

            s_proj_options = [args.s_proj]

            for s_proj in s_proj_options:
                output_dir = f"non_iid/{cond_str}/{f_0}_{s_proj}"

                if os.path.exists(f"{output_dir}/trajectory_{repeat_id}.pt"):
                    try:
                        torch.load(f"{output_dir}/trajectory_{repeat_id}.pt")
                        has_trajectory = True
                    except:
                        has_trajectory = False
                else:
                    has_trajectory = False
                if has_trajectory:
                    print(f"{output_dir}/trajectory_{repeat_id}.pt already exists")
                    continue
                os.makedirs(f"{output_dir}/{repeat_id}", exist_ok=True)
                seed_everything(repeat_id)
                out_t, trajectory = generate_image(
                    diffusion_model=base.diffusion_model,
                    vae=vae,
                    latent_fmt=latent_fmt,
                    sampling=sampling,
                    image_L=args.image_L,
                    steps=args.steps,
                    device=device,
                    cond=cond,
                    seed=repeat_id,
                    B=args.B,
                    diversity_type="dpp",
                    f_0=f_0,
                    s_proj=s_proj,
                )
                for i in range(args.B):
                    image = torch.clamp((out_t[i] + 1.0) / 2.0, min=0.0, max=1.0)
                    decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
                    decoded_np = decoded_np.astype(np.uint8)
                    out_image = Image.fromarray(decoded_np)
                    out_image.save(f"{output_dir}/{repeat_id}/{i}.png")
                torch.save(trajectory, f"{output_dir}/trajectory_{repeat_id}.pt")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
