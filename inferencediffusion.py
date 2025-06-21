import marimo

__generated_with = "0.13.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import deepinv
    import torch
    from torchvision import transforms
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    return deepinv, mo, np, plt, torch, transforms


@app.cell
def _(torch, transforms):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 48
    image_size = 32

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),
    ])

    print(f"Using device: {device}")
    return device, image_size


@app.cell
def _(deepinv, device, mo, torch):
    try:
        # Load the trained model
        model = deepinv.models.DiffUNet(
            in_channels=1, 
            out_channels=1, 
            pretrained=None
        ).to(device)

        # Load trained weights
        model.load_state_dict(torch.load("trained_diffusion_model.pth", map_location=device))
        model.eval()

        mo.md("✅ **Trained model loaded successfully!**")
    except FileNotFoundError:
        mo.md("❌ **Error:** `trained_diffusion_model.pth` not found. Please train the model first.")
        model = None
    return (model,)


@app.cell
def _():
    # Diffusion parameters (same as training)
    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000
    return beta_end, beta_start, timesteps


@app.cell
def _(beta_end, beta_start, device, timesteps, torch):
    # Compute diffusion schedule (same as training)
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return alphas, betas, sqrt_alphas_cumprod


@app.cell
def _(mo):
    # Interactive controls for generation
    mo.md("## 🎛️ Generation Controls")
    return


@app.cell
def _(mo):
    num_samples_slider = mo.ui.slider(
        start=1, stop=25, value=9, step=1,
        label="Number of samples to generate:"
    )

    show_process_checkbox = mo.ui.checkbox(
        value=False,
        label="Show generation process (slower)"
    )

    sampling_steps_slider = mo.ui.slider(
        start=10, stop=1000, value=50, step=10,
        label="Sampling steps (fewer = faster, more = better quality):"
    )

    mo.hstack([
        mo.vstack([num_samples_slider, show_process_checkbox]),
        sampling_steps_slider
    ])
    return num_samples_slider, sampling_steps_slider, show_process_checkbox


@app.cell
def _(
    alphas,
    betas,
    device,
    image_size,
    model,
    sqrt_alphas_cumprod,
    timesteps,
    torch,
):
    def generate_samples(num_samples=9, save_intermediate=False, skip_steps=50):
        """Generate samples using DDPM sampling with optional step skipping"""
        if model is None:
            return None, []

        model.eval()

        with torch.no_grad():
            # Start with pure noise
            x = torch.randn(num_samples, 1, image_size, image_size, device=device)

            intermediate_steps = []
            steps_to_run = list(range(0, timesteps, skip_steps))[::-1]  # Reverse and skip

            # Reverse diffusion process
            for i, t in enumerate(steps_to_run):
                t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

                # Predict noise at timestep t
                predicted_noise = model(x, t_batch, type_t="timestep")

                # DDPM reverse step
                alpha_t = alphas[t]
                alpha_cumprod_t = sqrt_alphas_cumprod[t]
                beta_t = betas[t]

                # Mean of reverse distribution
                mean = (1 / torch.sqrt(alpha_t)) * (
                    x - (beta_t / torch.sqrt(1 - alpha_cumprod_t**2)) * predicted_noise
                )

                if t > 0:
                    # Add noise (except for final step)
                    posterior_variance = beta_t * skip_steps  # Adjust for skipped steps
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(posterior_variance) * noise
                else:
                    x = mean

                # Save intermediate steps for visualization
                if save_intermediate and i % max(1, len(steps_to_run) // 8) == 0:
                    intermediate_steps.append(x.clone())

        return x, intermediate_steps
    return (generate_samples,)


@app.cell
def _(np, plt):
    def create_sample_plot(samples, title="Generated Samples"):
        """Create matplotlib figure for samples"""
        if samples is None:
            return None

        num_samples = samples.shape[0]
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))

        # Handle single subplot case
        if num_samples == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i in range(num_samples):
            # Convert to numpy and denormalize
            img = samples[i].cpu().squeeze().numpy()
            img = np.clip(img, 0, 1)

            axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[i].axis('off')
            axes[i].set_title(f'#{i+1}', fontsize=10)

        # Hide empty subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')

        plt.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout()
        return fig

    def create_process_plot(intermediate_steps, sample_idx=0):
        """Create matplotlib figure showing generation process"""
        if not intermediate_steps:
            return None

        num_steps = len(intermediate_steps)
        fig, axes = plt.subplots(1, num_steps, figsize=(2.5*num_steps, 2.5))

        if num_steps == 1:
            axes = [axes]

        for i, step in enumerate(intermediate_steps):
            img = step[sample_idx].cpu().squeeze().numpy()
            img = np.clip(img, 0, 1)

            axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[i].axis('off')
            axes[i].set_title(f'Step {i+1}', fontsize=10)

        plt.suptitle(f'Generation Process (Sample #{sample_idx+1})', fontsize=12)
        plt.tight_layout()
        return fig

    return create_process_plot, create_sample_plot


@app.cell
def _(mo):
    generate_button = mo.ui.button(
        label="🎨 Generate New Samples",
        kind="success"
    )
    generate_button
    return (generate_button,)


@app.cell
def _(
    create_process_plot,
    create_sample_plot,
    generate_button,
    generate_samples,
    mo,
    model,
    num_samples_slider,
    sampling_steps_slider,
    show_process_checkbox,
):
    # Generate samples when button is clicked
    if generate_button.value is not None and generate_button.value > 0 and model is not None:

        # Show loading message
        with mo.status.spinner(title="Generating samples..."):
            generated_images, intermediate = generate_samples(
                num_samples=num_samples_slider.value,
                save_intermediate=show_process_checkbox.value,
                skip_steps=sampling_steps_slider.value
            )

        if generated_images is not None:
            # Create plots
            samples_fig = create_sample_plot(generated_images, "🎯 Generated MNIST Digits")

            # Display results
            results = [mo.as_html(samples_fig)]

            # Add process visualization if requested
            if show_process_checkbox.value and intermediate:
                process_fig = create_process_plot(intermediate, sample_idx=0)
                if process_fig:
                    results.append(mo.md("### 🔄 Generation Process"))
                    results.append(mo.as_html(process_fig))

            mo.vstack(results)
        else:
            mo.md("❌ Failed to generate samples")
    elif model is None:
        mo.md("⚠️ Please load a trained model first")
    else:
        # This case handles when generate_button.value is None or 0,
        # and model is not None.
        # It implies the button hasn't been clicked yet or was reset.
        mo.md("👆 Click the button above to generate samples")
    return (generated_images,)


@app.cell
def _(mo):
    mo.md(
        """
    ---
    ## 💾 Save Options
    """
    )
    return


@app.cell
def _(mo):
    save_tensor_button = mo.ui.button(label="💾 Save as Tensor (.pt)", kind="neutral")
    save_image_button = mo.ui.button(label="🖼️ Save as Image (.png)", kind="neutral")

    mo.hstack([save_tensor_button, save_image_button])
    return save_image_button, save_tensor_button


@app.cell
def _(generated_images, mo, np, save_image_button, save_tensor_button, torch):
    if save_tensor_button.value is not None and save_tensor_button.value > 0:
        try:
            torch.save(generated_images, "generated_samples.pt")
            mo.md("✅ Samples saved as `generated_samples.pt`")
        except Exception as e:
            mo.md(f"❌ Error saving tensor file: {e}")

    if save_image_button.value is not None and save_image_button.value > 0:
        try:
            from torchvision.utils import save_image
            save_image(generated_images, "generated_grid.png", nrow=int(np.sqrt(generated_images.shape[0])), normalize=True, pad_value=1)
            mo.md("✅ Samples saved as `generated_grid.png`")
        except Exception as e:
            mo.md(f"❌ Error saving image file: {e}")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ---
    ## 📊 Model Info

    - **Architecture**: DiffUNet (DeepInv)
    - **Image Size**: 32x32 pixels
    - **Channels**: 1 (Grayscale)
    - **Timesteps**: 1000
    - **Dataset**: MNIST digits
    """
    )
    return


if __name__ == "__main__":
    app.run()
