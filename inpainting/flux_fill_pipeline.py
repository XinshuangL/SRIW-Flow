# Copyright 2025 Black Forest Labs and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def normalize_K(K: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Normalize the distance matrix K by the median of the upper triangle."""
    upper_tri_mask = torch.triu(torch.ones_like(K, dtype=torch.bool), diagonal=1)
    K_upper = K[upper_tri_mask]
    median_val = torch.median(K_upper).detach() + eps
    return K / median_val, K_upper / median_val


def compute_K(x: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances with median-upper normalization.

    Args:
        x: (B, D) matrix, each row is a flattened sample vector.
    Returns:
        K: (B, B) distance matrix normalized by median of upper triangle.
        K_upper: (num_pairs,) vector of the normalized upper-triangular distances.
    """
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    K = x_norm + x_norm.T - 2 * x @ x.T
    return normalize_K(K)


def _flat_to_map(x_flat: torch.Tensor, C: int):
    """
    Restore a packed or flattened latent vector back to a spatial feature map.

    Inputs:
        x_flat: (B, P*C_pack) or (B, P, C_pack), P = (H_lat/2)*(W_lat/2), C_pack = 4*C
        C:      number of original latent channels
    Returns:
        x_map: (B, C, H_lat, W_lat)
    """
    B = x_flat.shape[0]
    C_pack = 4 * C
    x_flat = x_flat.view(B, -1, C_pack)
    P = x_flat.shape[1]
    H_lat = int((P) ** 0.5 * 2)
    W_lat = int((P) ** 0.5 * 2)

    x_map = x_flat.view(B, H_lat // 2, W_lat // 2, C, 2, 2)
    x_map = x_map.permute(0, 3, 1, 4, 2, 5).contiguous()
    x_map = x_map.view(B, C, H_lat, W_lat)

    return x_map


def logdet_psd_cholesky(K: torch.Tensor, jitter: float = 1e-3) -> torch.Tensor:
    B = K.size(0)
    old_dtype = K.dtype
    K = K.to(dtype=torch.float32)
    I = torch.eye(B, dtype=K.dtype, device=K.device)
    L, info = torch.linalg.cholesky_ex(K + jitter * I)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
    return logdet.to(dtype=old_dtype)


def mask_extraction(f_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert mask.dim() == 4, "mask should be (1, 1, H, W)"

    _, _, H, W = f_map.shape
    if mask.shape[-2:] != (H, W):
        raise ValueError(f"Spatial size mismatch: feature_map {f_map.shape[-2:]}, mask {mask.shape[-2:]}")

    f_map_flat = f_map.view(f_map.shape[0], f_map.shape[1], -1)
    mask_flat = mask.view(1, -1)
    idx = mask_flat.squeeze(0) > 0
    selected = f_map_flat[:, :, idx]

    return selected


def _compute_diversity_and_force(
    diversity_name: str,
    latents: torch.Tensor,
    mask: torch.Tensor,
    abs_dt: float,
    *,
    velocity: Optional[torch.Tensor] = None,
    sigma: Optional[float] = None,
    f_0: float = 1.0,
    s_proj: str = "soft",
) -> (torch.Tensor, torch.Tensor, Dict):
    """
    Unified computation of both diversity scalar and its gradient ("force").
    The gradient is masked (applies only where mask==1) and post-processed by
    normalization, time scheduling, and global scaling.
    """
    old_dtype = latents.dtype
    latents = latents.to(torch.float32)
    mask = mask.to(torch.float32)
    velocity = velocity.to(torch.float32)

    C = latents.shape[2] // 4
    velocity = _flat_to_map(velocity, C)
    latents = _flat_to_map(latents, C)
    B, C, H, W = latents.shape

    latents_extracted = mask_extraction(latents, mask)
    velocity_extracted = mask_extraction(velocity, mask)

    latents_vector = latents_extracted.mean(dim=2)
    velocity_vector = velocity_extracted.mean(dim=2)
    score_vector = (velocity_vector * (1 - sigma) - latents_vector) / sigma

    if B < 2:
        return latents.new_zeros(()), torch.zeros_like(latents)

    with torch.enable_grad():
        l_req = latents_vector.detach().requires_grad_(True)
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

    combined_force_NCHW = combined_force.view(B, C, 1, 1).repeat(1, 1, H, W) * mask
    combined_force_token = FluxFillPipeline._pack_latents(combined_force_NCHW, B, C, H, W)

    to_save = {
        'latents_vector': latents_vector.detach().cpu(),
        'velocity_vector': velocity_vector.detach().cpu(),
        'score_vector': score_vector.detach().cpu(),
        'combined_force': combined_force.detach().cpu(),
        'diversity': diversity.detach().cpu(),
        'sigma': sigma,
        'abs_dt': abs_dt,
    }

    return diversity.detach().to(old_dtype), combined_force_token.detach().to(old_dtype), to_save


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` and returns (timesteps, num_inference_steps).
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"{scheduler.__class__}.set_timesteps does not support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"{scheduler.__class__}.set_timesteps does not support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class FluxFillPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    """
    Flux Fill pipeline for image inpainting/outpainting (single-GPU, simplified).

    Reference: https://blackforestlabs.ai/flux-1-tools/
    """

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2, vae_latent_channels=self.latent_channels
        )
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2,
            vae_latent_channels=self.latent_channels,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128

    def remove_text_models(self):
        del self.text_encoder
        del self.text_encoder_2
        self.text_encoder = None
        self.text_encoder_2 = None

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        num_channels_latents,
        num_images_per_prompt,
        height,
        width,
        dtype,
        device,
        generator,
    ):
        # 1. calculate the height and width of the latents
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        # 2. encode the masked image
        if masked_image.shape[1] == num_channels_latents:
            masked_image_latents = masked_image
        else:
            masked_image_latents = retrieve_latents(self.vae.encode(masked_image), generator=generator)

        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        # 3. duplicate mask and masked_image_latents
        batch_size = batch_size * num_images_per_prompt
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        # 4. pack the masked_image_latents
        masked_image_latents = self._pack_latents(
            masked_image_latents,
            batch_size,
            num_channels_latents,
            height,
            width,
        )

        # 5.resize mask to latents shape we concatenate the mask to the latents
        mask = mask[:, 0, :, :]
        mask = mask.view(
            batch_size, height, self.vae_scale_factor, width, self.vae_scale_factor
        )
        mask = mask.permute(0, 2, 4, 1, 3)
        mask = mask.reshape(
            batch_size, self.vae_scale_factor * self.vae_scale_factor, height, width
        )

        # 6. pack the mask
        mask = self._pack_latents(
            mask,
            batch_size,
            self.vae_scale_factor * self.vae_scale_factor,
            height,
            width,
        )
        mask = mask.to(device=device, dtype=dtype)

        return mask, masked_image_latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents

    def get_timesteps(self, num_inference_steps, strength, device):
        """Subset the scheduler timesteps given a strength (img2img convention)."""
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)
        return timesteps, num_inference_steps - t_start

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    def prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        if latents is not None:
            return latents.to(device=device, dtype=dtype), latent_image_ids

        image = image.to(device=device, dtype=dtype)
        if image.shape[1] != self.latent_channels:
            image_latents = self._encode_vae_image(image=image, generator=generator)
        else:
            image_latents = image
        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self.scheduler.scale_noise(image_latents, timestep, noise)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        return latents, latent_image_ids

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[torch.FloatTensor] = None,
        mask_image: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        diversity_type: str = "none",
        diversity_f_0: float = 0.0,
        diversity_s_proj: str = "soft",
    ):
        """
        Run inpainting with N parallel samples and diversity force applied inside the mask.
        Returns:
            List of output images (length = num_images_per_prompt).
        """
        device = self._execution_device

        self._joint_attention_kwargs = joint_attention_kwargs

        pipe_dtype = self.transformer.dtype

        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=pipe_dtype, device=device)

        batch_size = 1

        prompt_embeddings = torch.load('prompt_embeddings.pt')
        prompt_embeds = prompt_embeddings['prompt_embeds'].to(device=device, dtype=pipe_dtype)
        pooled_prompt_embeds = prompt_embeddings['pooled_prompt_embeds'].to(device=device, dtype=pipe_dtype)
        text_ids = prompt_embeddings['text_ids'].to(device)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, 1.0, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        num_channels_latents = self.vae.config.latent_channels
        latents, latent_image_ids = self.prepare_latents(
            init_image,
            latent_timestep,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        mask_image = self.mask_processor.preprocess(mask_image, height=height, width=width)
        mask_image = mask_image.to(device=device, dtype=prompt_embeds.dtype)
        masked_image = init_image * (1 - mask_image)
        masked_image = masked_image.to(device=device, dtype=prompt_embeds.dtype)

        latent_mask_2d = F.interpolate(mask_image, size=(height // self.vae_scale_factor, width // self.vae_scale_factor), mode='nearest')

        height, width = init_image.shape[-2:]
        mask, masked_image_latents = self.prepare_mask_latents(
            mask_image,
            masked_image,
            batch_size,
            num_channels_latents,
            num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )
        masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)

        assert self.transformer.config.guidance_embeds
        guidance = torch.full([1], 1.0, device=device, dtype=prompt_embeds.dtype).expand(latents.shape[0])

        to_save_list = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=torch.cat((latents, masked_image_latents), dim=2),
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self._joint_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.to(device=latents.device, dtype=latents.dtype)

                sigma_i = float(t / 1000)
                if i < len(timesteps) - 1:
                    sigma_ip1 = float(timesteps[i + 1] / 1000)
                else:
                    sigma_ip1 = 0.0
                dt = sigma_ip1 - sigma_i

                my_dt = -dt
                my_velocity = -noise_pred


                if diversity_type != "none":
                    _, force, to_save = _compute_diversity_and_force(
                        diversity_name=diversity_type,
                        latents=latents,
                        mask=latent_mask_2d,
                        abs_dt=abs(my_dt),
                        velocity=my_velocity,
                        sigma=sigma_i,
                        f_0=diversity_f_0,
                        s_proj=diversity_s_proj,
                    )
                else:
                    force = torch.zeros_like(latents)
                    to_save = None

                if to_save is not None:
                    to_save_list.append(to_save)

                latents = latents + my_velocity * my_dt + force * abs(my_dt)

                cur_samples = latents

                progress_bar.update()

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        images = self.vae.decode(latents, return_dict=False)[0]
        image_list = [self.image_processor.postprocess(image.unsqueeze(0), output_type=output_type)[0] for image in images]

        self.maybe_free_model_hooks()

        return image_list, {
            'to_save_list': to_save_list,
            'samples': cur_samples.detach().cpu(),
        }
