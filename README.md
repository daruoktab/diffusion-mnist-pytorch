# MNIST diffusion (DDPM) with PyTorch and DeepInv

Small portfolio project: train a noise-predicting U-Net on MNIST with a linear beta schedule (DDPM-style), save weights, and generate digits interactively in a second marimo app.

## Structure

```
diffusion-mnist-pytorch/
├── trainingdiffusion.py    # Marimo: train DiffUNet, plots, save checkpoint
├── inferencediffusion.py   # Marimo: load weights, DDPM sampling, UI
├── environment.yml         # Conda: Python 3.12
├── requirements.txt        # pip / uv
├── LICENSE                 # MIT
├── README.md
└── data/                   # MNIST (downloaded on first run; gitignored)
```

Default checkpoint path: `trained_diffusion_model.pth` (used by both apps).

## Environment

Conda for Python 3.12; install dependencies into that env (e.g. with `uv pip`):

```bash
conda env create -f environment.yml
conda activate dmnist
uv pip install -r requirements.txt
```

Without Conda: `uv venv --python 3.12 .venv`, activate it, then `uv pip install -r requirements.txt`.

## Run

Training (interactive editor):

```bash
marimo edit trainingdiffusion.py
```

Or one-shot: `uv run marimo run trainingdiffusion.py`

Inference / sampling:

```bash
marimo edit inferencediffusion.py
```

## Model and training

| Item | Value |
|------|--------|
| Architecture | `deepinv.models.DiffUNet`, 1→1 channels |
| Input size | 32×32 grayscale (resize from MNIST) |
| Schedule | Linear β from 1e−4 to 0.02, T = 1000 |
| Objective | MSE on predicted noise |
| Optimizer | Adam, lr = 1e−4 |

**VRAM:** Increase or decrease mainly via `batch_size` and `image_size` in `trainingdiffusion.py` (see the configuration cell).

## Forward noising (training)

```python
noisy_imgs = (
    sqrt_alphas_cumprod[t, None, None, None] * imgs
    + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
)
```

## References

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [DeepInv](https://deepinv.github.io/)
- [marimo](https://marimo.io/)

## License

[MIT](LICENSE). You may replace the copyright line in `LICENSE` with your name if this is a personal repo.
