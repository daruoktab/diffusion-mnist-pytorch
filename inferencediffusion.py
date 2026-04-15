# Dependencies: install from repo root (`uv pip install -r requirements.txt`)
# Validate: `uvx marimo check inferencediffusion.py`

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # MNIST diffusion — sampling

    Load `trained_diffusion_model.pth` from `trainingdiffusion.py`, run DDPM-style reverse diffusion, and inspect or save outputs.

    **Requires:** same image size (32×32) and schedule as training.
    """)
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
    from torchvision.utils import save_image

    return deepinv, np, plt, save_image, torch


@app.cell
def _(mo):
    mo.md("""
    ## Device
    """)
    return


@app.cell
def _(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    image_size = 32
    print(f"Device: {device}")
    return device, image_size


@app.cell
def _(mo):
    mo.md("""
    ## Load checkpoint
    """)
    return


@app.cell
def _(deepinv, device, mo, torch):
    model = None
    load_panel = mo.md("*Resolving checkpoint…*")
    try:
        model = deepinv.models.DiffUNet(
            in_channels=1,
            out_channels=1,
            pretrained=None,
        ).to(device)
        _ckpt = "trained_diffusion_model.pth"
        try:
            state = torch.load(_ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(_ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()
        load_panel = mo.md(
            "**Checkpoint loaded:** `trained_diffusion_model.pth` — ready to sample."
        )
    except FileNotFoundError:
        load_panel = mo.md(
            "**No checkpoint found.** Train with `trainingdiffusion.py` and ensure "
            "`trained_diffusion_model.pth` exists in this directory."
        )
    load_panel  # pyright: ignore[reportUnusedExpression] — marimo cell output
    return (model,)


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
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    # DDPM posterior variance σ̃²_t = β_t (1 - ᾱ_{t-1}) / (1 - ᾱ_t), with ᾱ_{-1} := 1
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1, device=device, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]]
    )
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp(
        min=1e-20
    )
    return alphas, betas, posterior_variance, sqrt_one_minus_alphas_cumprod


@app.cell
def _(mo):
    mo.md("""
    ## Generation controls
    """)
    return


@app.cell
def _(mo):
    num_samples_slider = mo.ui.slider(
        start=1, stop=25, value=9, step=1, label="Samples", full_width=True
    )
    show_process_checkbox = mo.ui.checkbox(
        value=False,
        label="Record intermediate steps (slower)",
    )
    generate_run = mo.ui.run_button(label="Generate samples")
    controls_ui = mo.vstack(
        [
            mo.md(
                "Sampling uses **full 1000-step DDPM** reverse diffusion (same schedule as training). "
                "Expect slower runs than a broken skip-stride shortcut; quality needs every step."
            ),
            mo.md(
                "In script mode (`marimo run`), generation runs automatically once after you open the app."
            ),
            num_samples_slider,
            show_process_checkbox,
            generate_run,
        ]
    )
    controls_ui  # pyright: ignore[reportUnusedExpression] — marimo cell output
    return generate_run, num_samples_slider, show_process_checkbox


@app.cell
def _(
    alphas,
    betas,
    device,
    image_size,
    model,
    posterior_variance,
    sqrt_one_minus_alphas_cumprod,
    timesteps,
    torch,
):
    def generate_samples(num_samples=9, save_intermediate=False):
        if model is None:
            return None, []

        model.eval()
        with torch.no_grad():
            x = torch.randn(num_samples, 1, image_size, image_size, device=device)
            intermediate_steps = []
            # Full DDPM reverse: one timestep at a time (Ho et al., Alg. 2).
            for step_i, t in enumerate(reversed(range(timesteps))):
                t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
                predicted_noise = model(x, t_batch, type_t="timestep")
                alpha_t = alphas[t]
                beta_t = betas[t]
                mean = (1.0 / torch.sqrt(alpha_t)) * (
                    x - (beta_t / sqrt_one_minus_alphas_cumprod[t]) * predicted_noise
                )
                if t > 0:
                    var = posterior_variance[t]
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(var) * noise
                else:
                    x = mean
                if save_intermediate and (
                    step_i % max(1, timesteps // 8) == 0 or t == 0
                ):
                    intermediate_steps.append(x.clone())
        return x, intermediate_steps

    return (generate_samples,)


@app.cell
def _(np, plt):
    def create_sample_plot(samples, title="Generated samples"):
        if samples is None:
            return None
        num_samples = samples.shape[0]
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        fig, ax_grid = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
        if num_samples == 1:
            ax_list = [ax_grid]
        else:
            ax_list = ax_grid.flatten()
        for i in range(num_samples):
            img = samples[i].cpu().squeeze().numpy()
            img = np.clip(img, 0, 1)
            ax_list[i].imshow(img, cmap="gray", vmin=0, vmax=1)
            ax_list[i].axis("off")
            ax_list[i].set_title(f"#{i + 1}", fontsize=10)
        for i in range(num_samples, len(ax_list)):
            ax_list[i].axis("off")
        plt.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout()
        return fig

    def create_process_plot(intermediate_steps, sample_idx=0):
        if not intermediate_steps:
            return None
        num_steps = len(intermediate_steps)
        fig, ax_row = plt.subplots(1, num_steps, figsize=(2.5 * num_steps, 2.8))
        if num_steps == 1:
            ax_iter = [ax_row]
        else:
            ax_iter = list(ax_row)
        for i, step in enumerate(intermediate_steps):
            img = step[sample_idx].cpu().squeeze().numpy()
            img = np.clip(img, 0, 1)
            ax_iter[i].imshow(img, cmap="gray", vmin=0, vmax=1)
            ax_iter[i].axis("off")
            ax_iter[i].set_title(f"Step {i + 1}", fontsize=10)
        plt.suptitle(f"Denoising (sample {sample_idx + 1})", fontsize=12, y=1.02)
        plt.tight_layout()
        return fig

    return create_process_plot, create_sample_plot


@app.cell
def _(mo):
    mo.md("""
    ## Sampling
    """)
    return


@app.cell
def _(
    generate_run,
    generate_samples,
    is_script_mode,
    mo,
    model,
    num_samples_slider,
    show_process_checkbox,
):
    generated_images = None
    intermediate_steps = []
    should_run = is_script_mode or (
        generate_run.value is not None and generate_run.value > 0
    )

    gen_panel = mo.md(
        "*Click **Generate samples**, or run this notebook with `marimo run` for a one-shot pass.*"
    )

    if model is None:
        gen_panel = mo.md("**No weights loaded** — fix the checkpoint path above first.")
    elif should_run:
        with mo.status.spinner(title="Sampling…"):
            generated_images, intermediate_steps = generate_samples(
                num_samples=num_samples_slider.value,
                save_intermediate=show_process_checkbox.value,
            )
        if generated_images is not None:
            gen_panel = mo.md(
                f"**Sampling finished** ({generated_images.shape[0]} images). "
                "See the figure(s) in the next cell."
            )
        else:
            gen_panel = mo.md("**Sampling failed** — model returned no tensor.")
    else:
        gen_panel = mo.md(
            "*Click **Generate samples** after choosing count and step stride.*"
        )

    gen_panel  # pyright: ignore[reportUnusedExpression] — marimo cell output
    return generated_images, intermediate_steps


@app.cell
def _(mo):
    mo.md("""
    ## Plots
    """)
    return


@app.cell
def _(
    create_process_plot,
    create_sample_plot,
    generated_images,
    intermediate_steps,
    mo,
    show_process_checkbox,
):
    # One matplotlib Figure: show it directly (same idea as trainingdiffusion.py).
    # Two figures: embed with mo.as_html inside mo.vstack (vstack expects HTML/UI mix).
    plot_panel = mo.md("*Generate samples above; plots appear here after a successful run.*")
    if generated_images is not None:
        samples_fig = create_sample_plot(generated_images, "Generated MNIST digits")
        if show_process_checkbox.value and intermediate_steps:
            process_fig = create_process_plot(intermediate_steps, sample_idx=0)
            if process_fig is not None:
                plot_panel = mo.vstack(
                    [
                        mo.as_html(samples_fig),
                        mo.md("### Denoising trajectory (first sample)"),
                        mo.as_html(process_fig),
                    ]
                )
            else:
                plot_panel = samples_fig
        else:
            plot_panel = samples_fig
    plot_panel  # pyright: ignore[reportUnusedExpression] — marimo cell output
    return


@app.cell
def _(mo):
    mo.md("""
    ## Save outputs
    """)
    return


@app.cell
def _(mo):
    save_tensor_button = mo.ui.button(label="Save tensor (.pt)", kind="neutral")
    save_image_button = mo.ui.button(label="Save image grid (.png)", kind="neutral")
    save_buttons_ui = mo.hstack([save_tensor_button, save_image_button])
    save_buttons_ui  # pyright: ignore[reportUnusedExpression] — marimo cell output
    return save_image_button, save_tensor_button


@app.cell
def _(
    generated_images,
    mo,
    np,
    save_image,
    save_image_button,
    save_tensor_button,
    torch,
):
    notes: list[str] = []

    if save_tensor_button.value is not None and save_tensor_button.value > 0:
        if generated_images is None:
            notes.append("**Tensor:** generate samples before saving.")
        else:
            try:
                torch.save(generated_images, "generated_samples.pt")
                notes.append("**Tensor:** wrote `generated_samples.pt`.")
            except OSError as e:
                notes.append(f"**Tensor:** could not save ({e}).")

    if save_image_button.value is not None and save_image_button.value > 0:
        if generated_images is None:
            notes.append("**Image:** generate samples before saving.")
        else:
            try:
                nrow = max(1, int(np.sqrt(generated_images.shape[0])))
                save_image(
                    generated_images,
                    "generated_grid.png",
                    nrow=nrow,
                    normalize=True,
                    pad_value=1,
                )
                notes.append("**Image:** wrote `generated_grid.png`.")
            except OSError as e:
                notes.append(f"**Image:** could not save ({e}).")

    save_panel = mo.md("\n\n".join(notes)) if notes else mo.md("*Save actions appear here.*")
    save_panel  # pyright: ignore[reportUnusedExpression] — marimo cell output
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Model notes

    | | |
    |--|--|
    | Architecture | `DiffUNet` (DeepInv) |
    | Spatial size | 32×32 grayscale |
    | Channels | 1 |
    | Training timesteps | 1000 (linear β schedule) |
    | Data | MNIST-style digits |
    """)
    return


if __name__ == "__main__":
    app.run()
