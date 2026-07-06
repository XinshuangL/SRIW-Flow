# Inpainting (FLUX.1-Fill-dev)

Diverse image inpainting: draw a **jointly-sampled set** of inpaintings for a masked region, with
score-based regularization keeping them on the data manifold. Backbone FLUX.1-Fill-dev, 512×512,
28 steps, fixed prompt `"a realistic photo"`, diversity strength `f_0 = 0.3` (the paper's λ).

Run every script from inside this folder. Sampling needs a GPU and the FLUX weights — `diffusers`
auto-downloads `black-forest-labs/FLUX.1-Fill-dev` on first run; the repo is gated, so accept its
license and run `huggingface-cli login` first.

## Run

Place your inputs at `images/0.jpg … 9.jpg` (`diverse_sampling.py` uses the first ten; `iid_gt.py`
runs on every image in the folder). The paper used 10 square images from the Open Images V7 test split.

```sh
cd inpainting
python iid_gt.py                          # IID reference inpaintings   -> images_iid/
python diverse_sampling.py --s_proj soft  # joint (non-IID) inpaintings -> images_pred_0.3/
```

`--s_proj` selects the score-regularization mode: `none`, `soft`, or `hard`.

## Files

- `flux_fill_pipeline.py` — FLUX inpainting pipeline with the diversity + score-regularization force inside the mask (imported by the scripts).
- `iid_gt.py` — IID sampler; `diverse_sampling.py` — joint diverse sampler (DPP).
- `mask.png`, `prompt_embeddings.pt` — shipped inputs; `prompt_embeddings.pt` holds the precomputed T5/CLIP embeddings for `"a realistic photo"`, so sampling runs text-free.

The coverage-radius metric reported in the paper is computed from the saved latents, as described there.
