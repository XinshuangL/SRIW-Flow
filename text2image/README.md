# Text-to-image (Stable Diffusion 3.5 Medium)

Diverse text-to-image joint sampling: for each prompt, generate a **jointly-sampled set** of images with score-based regularization keeping them on the data manifold. SD3.5-Medium (a flow-matching MMDiT) over six fixed prompts.

Run every script from inside this folder. Sampling needs a GPU and the SD3.5-Medium weights (gated — accept the Stability AI Community License and run `huggingface-cli login` before `download.py`).

## Run

```sh
cd text2image
python download.py                        # SD3.5-Medium checkpoint  -> models/
python generate_image_iid.py              # IID reference samples    -> iid/<prompt>/
python generate_image.py --s_proj soft    # joint (non-IID) samples  -> non_iid/<prompt>/0.1_<sr>/
```

`--s_proj` selects the score-regularization mode (`none` / `soft` / `hard`); `0.1` is the diversity strength λ. The six prompts (typos intentional) — `a fish`, `a realistic fish`, `a cat`, `a releastic cat`, `something outside`, `someone ahead` — ship precomputed in `conditionings/`, so `text_to_conditioning.py` is only needed to re-encode them (it requires the SD3 text encoders; see the top-level README).

## Files

- `generate_image.py` — joint diverse sampler (DPP × SR), saving per-step trajectories to `non_iid/.../trajectory_<id>.pt`.
- `generate_image_iid.py` — IID sampler; `text_to_conditioning.py`, `download.py` — conditioning / weight prep.
- `sd3_impls.py`, `sd3_infer.py`, `mmditx.py`, `other_impls.py` — vendored SD3 reference implementation (Stability AI `sd3-ref`, MIT; parts of `other_impls.py` are Apache-2.0, from Hugging Face `transformers`).
- `conditionings/<prompt>.pt` — shipped precomputed conditionings for the six prompts.

The coverage-radius metric reported in the paper is computed from the saved latents/trajectories, as described there.
