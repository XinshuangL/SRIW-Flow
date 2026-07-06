# Score-Regularized Joint Sampling with Importance Weights for Flow Matching

[Project page](https://XinshuangL.github.io/SRIW-Flow) · [Paper](https://arxiv.org/abs/2511.17812) (UAI 2026)

Given a **pre-trained** flow-matching model, we draw a *set* of samples jointly rather than independently — covering more of the distribution — and use the model's own score to keep the added diversity on the data manifold. A trajectory-based importance-weight estimator then debiases expectations over these non-IID sets.

## Installation

```sh
conda create -n sriw-flow python=3.13 && conda activate sriw-flow
pip install torch numpy scipy tqdm matplotlib jupyter            # gaussian (CPU notebook)
pip install safetensors huggingface_hub pillow einops transformers   # text-to-image
pip install diffusers transformers opencv-python                # inpainting
```

Tested with Python 3.13, torch 2.8, transformers 5.3, diffusers 0.36. Re-encoding prompts also needs
`sentencepiece protobuf`.

## Models

Weights are gated — accept the license on Hugging Face and run `huggingface-cli login` once.

| Model | Hugging Face repo | Place at |
|---|---|---|
| SD 3.5 Medium (~4.8 GB) | `stabilityai/stable-diffusion-3.5-medium` | `text2image/models/sd3.5_medium.safetensors` (via `download.py`) |
| SD3 text encoders (only to re-encode prompts) | same distribution | `text2image/models/{clip_l,clip_g,t5xxl}.safetensors` |
| FLUX.1-Fill-dev | `black-forest-labs/FLUX.1-Fill-dev` | auto-downloaded by `diffusers` on first run |

## Quickstart

**Gaussian** — the whole method on a controlled mixture (CPU, no weights, no data):

```sh
cd gaussian && jupyter notebook GMM_core.ipynb
```

**Text-to-image** (SD 3.5 Medium) — the six prompts ship precomputed in `text2image/conditionings/`:

```sh
cd text2image
python download.py                        # SD3.5-Medium checkpoint
python generate_image_iid.py              # IID reference samples
python generate_image.py --s_proj soft    # joint (non-IID) samples;  --s_proj none|soft|hard
```

**Inpainting** (FLUX.1-Fill-dev) — supply your own images at `inpainting/images/0.jpg … 9.jpg`
(the paper used 10 from the Open Images V7 test split):

```sh
cd inpainting
python iid_gt.py                          # IID reference inpaintings
python diverse_sampling.py --s_proj soft  # joint (non-IID) inpaintings;  --s_proj none|soft|hard
```

## Citation

```bibtex
@inproceedings{liu2026score,
  title     = {Score-Regularized Joint Sampling with Importance Weights for Flow Matching},
  author    = {Liu, Xinshuang and Li, Runfa Blark and Wei, Shaoxiu and Nguyen, Truong},
  booktitle = {Conference on Uncertainty in Artificial Intelligence (UAI)},
  year      = {2026}
}
```

## License

Our code is released under the [MIT License](LICENSE). Vendored components keep their own licenses:
the SD3 reference implementation (`text2image/{mmditx,other_impls,sd3_impls,sd3_infer}.py`) is from
Stability AI's MIT-licensed [`sd3-ref`](https://github.com/Stability-AI/sd3-ref) (parts of
`other_impls.py` are Apache-2.0, from Hugging Face `transformers`); the FLUX pipeline
(`inpainting/flux_fill_pipeline.py`) is Apache-2.0 (Black Forest Labs / Hugging Face). Downloaded
model weights carry their own terms — SD3.5-Medium under the Stability AI Community License and
FLUX.1-Fill-dev under the FLUX.1 [dev] Non-Commercial License.
