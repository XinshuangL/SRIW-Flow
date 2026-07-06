# Gaussian mixture

A single self-contained notebook, **`GMM_core.ipynb`**, runs the whole method on Gaussian mixtures with known ground truth — a sparse 8-D mixture for the diversity/quality study and a 2-D mixture for importance-weight estimation. It defines the diversity objectives (DPP and Harmonic DPP) with score-based regularization (`none` / `soft` / `hard`), draws the joint (non-IID) sample sets, and learns a residual velocity for the trajectory-based importance weights.

## Run

```sh
cd gaussian
jupyter notebook GMM_core.ipynb    # Run all
```

CPU only — the mixtures and flow-matching dynamics are defined inline (no pretrained weights). Requires `torch numpy scipy tqdm matplotlib jupyter`.
