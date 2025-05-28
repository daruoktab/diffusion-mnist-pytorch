import marimo

__generated_with = "0.13.13"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
    # 🎨 Diffusion Model for MNIST Generation

    This notebook demonstrates a trained diffusion model for generating MNIST digits. 
    The model uses a U-Net architecture with diffusion process to generate realistic handwritten digits.

    ## Overview
    - **Model**: DiffUNet (U-Net for diffusion)
    - **Dataset**: MNIST (28x28 grayscale digits)
    - **Process**: Forward diffusion adds noise, reverse diffusion generates images
    - **Timesteps**: 1000 steps for the diffusion process
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""## 📦 Import Libraries and Setup""")
    return


@app.cell
def _():
    import deepinv
    import torch
    from torchvision import datasets, transforms
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    return datasets, deepinv, np, os, plt, torch, transforms


@app.cell
def _(mo):
    mo.md("""## ⚙️ Configuration and Device Setup""")
    return


@app.cell
def _(torch, transforms):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 48
    image_size = 32

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,)),
        ]
    )

    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}x{image_size}")
    return batch_size, device, transform


@app.cell
def _(mo):
    mo.md("""## 📊 Data Loading""")
    return


@app.cell
def _(batch_size, datasets, torch, transform):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    return (train_loader,)


@app.cell
def _(mo):
    mo.md("""## 🏗️ Model Architecture and Training Parameters""")
    return


@app.cell
def _(deepinv, device, torch):
    lr = 1e-4
    epochs = 5

    model = deepinv.models.DiffUNet(
        in_channels=1, 
        out_channels=1, 
        pretrained=None).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return (model,)


@app.cell
def _(mo):
    mo.md("""## 🔄 Diffusion Process Parameters""")
    return


@app.cell
def _():
    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000

    print(f"Beta start: {beta_start}")
    print(f"Beta end: {beta_end}")
    print(f"Total timesteps: {timesteps}")
    return beta_end, beta_start, timesteps


@app.cell
def _(beta_end, beta_start, device, timesteps, torch):
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    print("Diffusion schedule computed successfully!")
    return (
        alphas,
        alphas_cumprod,
        betas,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Training Code 

    The code below shows the training loop 
    that was used to train the diffusion model:
    """
    )
    return


@app.cell
def _():
    # TRAINING CODE - ALREADY COMPLETED, COMMENTED OUT TO PREVENT RE-TRAINING

    # for epoch in range(epochs):
    #     print(f"Epoch {epoch + 1}/{epochs}")
    #     model.train()
    #     epoch_loss = 0.0
    #     num_batches = 0
    #     
    #     for data, _ in train_loader:
    #         imgs = data.to(device)
    #         noise = torch.randn_like(imgs)
    #         t = torch.randint(0, timesteps, (imgs.size(0),), device=device)
    #
    #         noisy_imgs = (
    #             sqrt_alphas_cumprod[t, None, None, None] * imgs +
    #             sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
    #         )
    #         
    #         optimizer.zero_grad()
    #         estimated_noise = model(noisy_imgs, t, type_t="timestep")
    #         loss = mse(estimated_noise, noise)
    #         loss.backward()
    #         optimizer.step()
    #         
    #         epoch_loss += loss.item()
    #         num_batches += 1
    #     
    #     avg_loss = epoch_loss / num_batches
    #     print(f"Average loss for epoch {epoch + 1}: {avg_loss:.6f}")

    print("Training completed")
    return


@app.cell
def _(mo):
    mo.md("""## 💾 Model Loading/Saving""")
    return


@app.cell
def _(device, model, os, torch):
    model_path = "trained_diffusion_model.pth"

    # Load the trained model if it exists
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ Loaded trained model from {model_path}")
        model_loaded = True
    else:
        print(f"❌ Model file {model_path} not found. Please ensure the model is trained and saved.")
        model_loaded = False

    # Uncomment to save the model after training:
    # torch.save(model.state_dict(), model_path)
    # print(f"Model saved to {model_path}")

    return (model_loaded,)


@app.cell
def _(mo):
    mo.md("""## 🎲 Interactive Generation Controls""")
    return


@app.cell
def _(mo):
    # Interactive controls for generation
    num_samples_slider = mo.ui.slider(1, 16, value=8, label="Number of samples to generate")
    generation_steps_slider = mo.ui.slider(50, 1000, value=200, step=50, label="Generation steps")
    random_seed_input = mo.ui.number(value=42, label="Random seed (for reproducibility)")

    mo.md(f"""
    ### Generation Parameters

    {num_samples_slider}

    {generation_steps_slider}

    {random_seed_input}
    """)
    return generation_steps_slider, num_samples_slider, random_seed_input


@app.cell
def _(
    generation_steps_slider,
    mo,
    model_loaded,
    num_samples_slider,
    random_seed_input,
):
    generate_button = mo.ui.button(
        label="🎨 Generate Images", 
        disabled=not model_loaded,
        tooltip="Generate new MNIST digits using the diffusion model"
    )

    if not model_loaded:
        mo.md("⚠️ **Model not loaded!** Please ensure the trained model file exists.")
    else:
        mo.md(f"""
        **Generation Settings:**
        - Samples: {num_samples_slider.value}
        - Steps: {generation_steps_slider.value}
        - Seed: {random_seed_input.value}

        {generate_button}
        """)
    return (generate_button,)


@app.cell
def _(
    alphas,
    betas,
    device,
    generate_button,
    generation_steps_slider,
    model,
    model_loaded,
    np,
    num_samples_slider,
    plt,
    random_seed_input,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    timesteps,
    torch,
):
    def generate_samples(model, num_samples, num_steps, seed=None):
        """Generate samples using the trained diffusion model"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        model.eval()
        with torch.no_grad():
            # Start with pure noise
            samples = torch.randn(num_samples, 1, 32, 32, device=device)

            # Reverse diffusion process
            step_size = timesteps // num_steps
            for _i in range(num_steps):
                t_idx = timesteps - 1 - _i * step_size
                t = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)

                # Predict noise
                predicted_noise = model(samples, t, type_t="timestep")

                # Remove noise
                alpha = alphas[t_idx]
                alpha_cumprod = sqrt_alphas_cumprod[t_idx] # This variable is assigned but not used in the original code
                beta = betas[t_idx]

                samples = (samples - beta * predicted_noise / sqrt_one_minus_alphas_cumprod[t_idx]) / torch.sqrt(alpha)

                # Add noise for non-final steps
                if _i < num_steps - 1:
                    noise = torch.randn_like(samples) * torch.sqrt(beta)
                    samples = samples + noise

        return samples

    # Generate images when button is clicked
    if generate_button.value:
        _num_samples_val = num_samples_slider.value
        _num_steps_val = generation_steps_slider.value
        _seed_val = random_seed_input.value

        if model_loaded: # Check if model is loaded, though button should be disabled if not
            _generated_samples_tensor = generate_samples(
                model,
                num_samples=_num_samples_val,
                num_steps=_num_steps_val,
                seed=_seed_val
            )
            _generated_images_np = _generated_samples_tensor.cpu().numpy()

            # Determine grid size for plotting
            # Max 4 columns, rows adjust accordingly
            _cols = min(_num_samples_val, 4)
            _rows = (_num_samples_val + _cols - 1) // _cols

            _fig_gen, _axes_gen = plt.subplots(_rows, _cols, figsize=(_cols * 3, _rows * 3.5))
            _fig_gen.suptitle(f"Generated Samples (Seed: {_seed_val}, Steps: {_num_steps_val})", fontsize=14)

            # Ensure _axes_gen is always a flat array for consistent indexing
            if _num_samples_val == 1:
                _axes_gen = [_axes_gen]
            else:
                _axes_gen = np.array(_axes_gen).flatten()

            for _plot_idx in range(_num_samples_val):
                _plot_img = _generated_images_np[_plot_idx].squeeze() # Remove channel dim for grayscale
                _axes_gen[_plot_idx].imshow(_plot_img, cmap='gray')
                _axes_gen[_plot_idx].set_title(f'Sample {_plot_idx+1}')
                _axes_gen[_plot_idx].axis('off')

            # Hide any unused subplots if _num_samples_val < _rows * _cols
            for _unused_ax_idx in range(_num_samples_val, len(_axes_gen)):
                _axes_gen[_unused_ax_idx].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
            plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""## 📈 Visualize Diffusion Schedule""")
    return


@app.cell
def _(alphas, alphas_cumprod, betas, np, plt, timesteps):
    # Visualize the diffusion schedule
    _fig_schedule, _axes_schedule = plt.subplots(1, 3, figsize=(15, 4))

    t_vals = np.arange(timesteps)

    # Beta schedule
    _axes_schedule[0].plot(t_vals, betas.cpu().numpy())
    _axes_schedule[0].set_title('Beta Schedule (Noise Rate)')
    _axes_schedule[0].set_xlabel('Timestep')
    _axes_schedule[0].set_ylabel('Beta')
    _axes_schedule[0].grid(True, alpha=0.3)

    # Alpha schedule
    _axes_schedule[1].plot(t_vals, alphas.cpu().numpy())
    _axes_schedule[1].set_title('Alpha Schedule (1 - Beta)')
    _axes_schedule[1].set_xlabel('Timestep')
    _axes_schedule[1].set_ylabel('Alpha')
    _axes_schedule[1].grid(True, alpha=0.3)

    # Alpha cumulative product
    _axes_schedule[2].plot(t_vals, alphas_cumprod.cpu().numpy())
    _axes_schedule[2].set_title('Alpha Cumulative Product')
    _axes_schedule[2].set_xlabel('Timestep')
    return


@app.cell
def _(mo):
    mo.md("""## 🔍 Explore Training Data""")
    return


@app.cell
def _(plt, train_loader):
    # Show some training examples
    _data_iter_train = iter(train_loader)
    _images_train, _labels_train = next(_data_iter_train)

    _fig_train_samples, _axes_train_samples = plt.subplots(2, 8, figsize=(16, 4))
    _axes_train_samples = _axes_train_samples.flatten()

    for i in range(16):
        _img_display = _images_train[i].squeeze().numpy()
        _axes_train_samples[i].imshow(_img_display, cmap='gray')
        _axes_train_samples[i].set_title(f'Label: {_labels_train[i].item()}')
        _axes_train_samples[i].axis('off')

    plt.tight_layout()
    plt.suptitle('Sample Training Images', y=1.02, fontsize=14) # Added fontsize for better visibility
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 🎯 Next Steps

    This interactive notebook allows you to:

    1. **Generate new images** using the trained diffusion model
    2. **Experiment with different parameters** (steps, samples, seeds)
    3. **Visualize the diffusion process** and training data
    4. **Understand the model architecture** and parameters

    ### Tips for Experimentation:
    - **More steps** = Higher quality but slower generation
    - **Different seeds** = Different random outputs
    - **Batch size** affects memory usage during training

    ### Model Performance:
    - The model was trained for **5 epochs** on MNIST
    - Uses **MSE loss** for noise prediction
    - **1000 timesteps** for full diffusion process
    """
    )
    return


if __name__ == "__main__":
    app.run()
