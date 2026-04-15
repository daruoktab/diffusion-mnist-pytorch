# Dependencies: install from repo root, e.g. `uv pip install -r requirements.txt`
# Validate: `uvx marimo check trainingdiffusion.py`
# marimo pair (AI + live notebook): https://docs.marimo.io/guides/generate_with_ai/marimo_pair/

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
# DDPM on MNIST (DeepInv DiffUNet)

Train a denoising diffusion model on MNIST: linear β schedule, noise prediction with MSE, and checkpoint export for sampling in `inferencediffusion.py`.

**Stack:** PyTorch, Torchvision, [DeepInv](https://deepinv.github.io/) `DiffUNet`, [marimo](https://marimo.io/).
"""
    )
    return


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _():
    import deepinv
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torchvision import datasets, transforms

    return datasets, deepinv, np, plt, torch, transforms


@app.cell
def _(mo):
    mo.md("## Configuration")
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Run training")
    mo.vstack(
        [
            mo.md(
                "**Training** — Click **Run training** in the UI, or use `marimo run trainingdiffusion.py` for a full non-interactive pass."
            ),
            mo.md(
                "Hyperparameters (`batch_size`, `epochs`, `learning_rate`, `image_size`) live in the next cell; edit there to tune VRAM and run length."
            ),
            train_button,
        ]
    )
    return (train_button,)


@app.cell
def _(torch, transforms):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # VRAM: raise or lower batch_size first; then image_size if needed.
    batch_size = 96
    epochs = 10
    learning_rate = 1e-4
    image_size = 32
    model_path = "trained_diffusion_model.pth"

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,)),
        ]
    )

    print(f"Device: {device}")
    print(
        f"Batch size: {batch_size} | Image: {image_size}x{image_size} | Epochs: {epochs} | lr: {learning_rate}"
    )
    return batch_size, device, epochs, learning_rate, model_path, transform


@app.cell
def _(mo):
    mo.md("## Data")
    return


@app.cell
def _(batch_size, datasets, device, torch, transform):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    return (train_loader,)


@app.cell
def _(mo):
    mo.md("## Model and optimizer")
    return


@app.cell
def _(deepinv, device, learning_rate, torch):
    model = deepinv.models.DiffUNet(
        in_channels=1,
        out_channels=1,
        pretrained=None,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse = torch.nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    return model, mse, optimizer


@app.cell
def _(mo):
    mo.md("## Diffusion schedule")
    return


@app.cell
def _():
    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000
    return beta_end, beta_start, timesteps


@app.cell
def _(beta_end, beta_start, device, timesteps, torch):
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return (
        alphas,
        alphas_cumprod,
        betas,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
    )


@app.cell
def _(mo):
    mo.md("### Noise schedule (linear β)")
    return


@app.cell
def _(alphas, alphas_cumprod, betas, np, plt, timesteps):
    fig_schedule, axes_schedule = plt.subplots(1, 3, figsize=(12, 3.5))
    t_vals = np.arange(timesteps)

    axes_schedule[0].plot(t_vals, betas.cpu().numpy(), color="#2563eb")
    axes_schedule[0].set_title("βₜ")
    axes_schedule[0].set_xlabel("t")
    axes_schedule[0].grid(True, alpha=0.3)

    axes_schedule[1].plot(t_vals, alphas.cpu().numpy(), color="#059669")
    axes_schedule[1].set_title("αₜ = 1 − βₜ")
    axes_schedule[1].set_xlabel("t")
    axes_schedule[1].grid(True, alpha=0.3)

    axes_schedule[2].plot(t_vals, alphas_cumprod.cpu().numpy(), color="#7c3aed")
    axes_schedule[2].set_title(r"$\bar{\alpha}_t$ (cumulative)")
    axes_schedule[2].set_xlabel("t")
    axes_schedule[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return (fig_schedule,)


@app.cell
def _(mo):
    mo.md("### Sample minibatch after preprocessing")
    return


@app.cell
def _(plt, train_loader):
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    fig_samples, axes_samples = plt.subplots(2, 8, figsize=(14, 3.8))
    axes_flat = axes_samples.flatten()
    for i in range(min(16, images.shape[0])):
        axes_flat[i].imshow(images[i].squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        axes_flat[i].set_title(str(labels[i].item()), fontsize=9)
        axes_flat[i].axis("off")
    for i in range(images.shape[0], len(axes_flat)):
        axes_flat[i].axis("off")
    plt.suptitle("MNIST (resized & normalized)", y=1.02, fontsize=12)
    plt.tight_layout()
    return (fig_samples,)


@app.cell
def _(mo):
    mo.md("## Training")
    return


@app.cell
def _(
    device,
    epochs,
    is_script_mode,
    mo,
    model,
    mse,
    optimizer,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    timesteps,
    torch,
    train_button,
    train_loader,
):
    loss_history: list[float] = []
    should_train = is_script_mode or (
        train_button.value is not None and train_button.value > 0
    )

    training_status_ui = mo.md(" ")
    if should_train:
        with mo.status.spinner(title="Training diffusion model…"):
            for epoch in range(epochs):
                model.train()
                running = 0.0
                n_batches = 0
                for data, _ in train_loader:
                    imgs = data.to(device)
                    noise = torch.randn_like(imgs)
                    t = torch.randint(0, timesteps, (imgs.size(0),), device=device)

                    noisy_imgs = (
                        sqrt_alphas_cumprod[t, None, None, None] * imgs
                        + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
                    )
                    optimizer.zero_grad()
                    estimated_noise = model(noisy_imgs, t, type_t="timestep")
                    loss = mse(estimated_noise, noise)
                    loss.backward()
                    optimizer.step()
                    running += loss.item()
                    n_batches += 1

                avg = running / max(n_batches, 1)
                loss_history.append(avg)
                print(f"Epoch {epoch + 1}/{epochs} — loss: {avg:.6f}")

        training_status_ui = mo.md(
            f"**Done.** Trained **{len(loss_history)}** epoch(s). "
            f"Final batch-averaged MSE: **{loss_history[-1]:.6f}**."
        )
    else:
        training_status_ui = mo.md(
            "*Click **Run training** (or run this file with `marimo run` / script mode).*"
        )
    training_status_ui
    return loss_history


@app.cell
def _(mo):
    mo.md("### Loss curve")
    return


@app.cell
def _(loss_history, mo, np, plt):
    loss_plot_out = mo.md("*Run training above to plot MSE vs epoch.*")
    if loss_history:
        fig_loss, ax_loss = plt.subplots(figsize=(8, 3.5))
        epochs_x = np.arange(1, len(loss_history) + 1)
        ax_loss.plot(
            epochs_x, loss_history, marker="o", color="#dc2626", linewidth=2, markersize=5
        )
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("MSE (noise prediction)")
        ax_loss.set_title("Training loss")
        ax_loss.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_plot_out = fig_loss
    loss_plot_out
    return


@app.cell
def _(loss_history, mo):
    metrics_out = mo.md("*No metrics table until training has run.*")
    if loss_history:
        rows = [{"epoch": i + 1, "loss_mse": round(v, 6)} for i, v in enumerate(loss_history)]
        metrics_out = mo.vstack([mo.md("### Per-epoch metrics"), mo.ui.table(rows)])
    metrics_out
    return


@app.cell
def _(mo):
    mo.md("## Save checkpoint")
    return


@app.cell
def _(loss_history, model, model_path, torch):
    if loss_history:
        torch.save(model.state_dict(), model_path)
        print(f"Saved state dict to {model_path}")
    else:
        print("Skip save: no training run in this session.")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
## Reference

- Ho et al., *Denoising Diffusion Probabilistic Models*, [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- After training, open `inferencediffusion.py` to sample from the saved weights.
"""
    )
    return


if __name__ == "__main__":
    app.run()
