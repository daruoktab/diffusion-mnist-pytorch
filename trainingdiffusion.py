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
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    return datasets, deepinv, np, os, plt, torch, transforms


@app.cell
def _(mo):
    mo.md("""## ⚙️ Configuration and Device Setup""")
    return


@app.cell
def _(torch, transforms):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
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

    model = deepinv.models.DiffUNet(
        in_channels=1, 
        out_channels=1, 
        pretrained=None).to(device)

    torch.optim.Adam(model.parameters(), lr=lr)
    torch.nn.MSELoss()

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
    torch.sqrt(alphas_cumprod)
    torch.sqrt(1.0 - alphas_cumprod)

    print("Diffusion schedule computed successfully!")
    return alphas, alphas_cumprod, betas


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
    # Training code (commented out since model is already trained)
    # for epoch in range(epochs):
    #     print(f"Epoch {epoch + 1}/{epochs}")
    #     model.train()
    #     epoch_loss = 0.0
    #     num_batches = 0

    #     for data, _ in train_loader:
    #         imgs = data.to(device)
    #         noise = torch.randn_like(imgs)
    #         t = torch.randint(0, timesteps, (imgs.size(0),), device=device)

    #         noisy_imgs = (
    #             sqrt_alphas_cumprod[t, None, None, None] * imgs +
    #             sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
    #         )

    #         optimizer.zero_grad()
    #         estimated_noise = model(noisy_imgs, t, type_t="timestep")
    #         loss = mse(estimated_noise, noise)
    #         loss.backward()
    #         optimizer.step()

    #         epoch_loss += loss.item()
    #         num_batches += 1

    #     avg_loss = epoch_loss / num_batches
    #     print(f"Average loss for epoch {epoch + 1}: {avg_loss:.6f}")

    # print("Training completed")
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
    else:
        print(f"❌ Model file {model_path} not found. Please ensure the model is trained and saved.")

    # Uncomment to save the model after training:
    # torch.save(model.state_dict(), model_path)
    # print(f"Model saved to {model_path}")

    return


@app.cell
def _(mo):
    mo.md("""## 📈 Visualize Diffusion Schedule""")
    return


@app.cell
def _(alphas, alphas_cumprod, betas, np, plt, timesteps):
    # Visualize the diffusion schedule
    fig_schedule, axes_schedule = plt.subplots(1, 3, figsize=(15, 4))

    t_vals = np.arange(timesteps)

    # Beta schedule
    axes_schedule[0].plot(t_vals, betas.cpu().numpy())
    axes_schedule[0].set_title('Beta Schedule (Noise Rate)')
    axes_schedule[0].set_xlabel('Timestep')
    axes_schedule[0].set_ylabel('Beta')
    axes_schedule[0].grid(True, alpha=0.3)

    # Alpha schedule
    axes_schedule[1].plot(t_vals, alphas.cpu().numpy())
    axes_schedule[1].set_title('Alpha Schedule (1 - Beta)')
    axes_schedule[1].set_xlabel('Timestep')
    axes_schedule[1].set_ylabel('Alpha')
    axes_schedule[1].grid(True, alpha=0.3)

    # Alpha cumulative product
    axes_schedule[2].plot(t_vals, alphas_cumprod.cpu().numpy())
    axes_schedule[2].set_title('Alpha Cumulative Product')
    axes_schedule[2].set_xlabel('Timestep')
    axes_schedule[2].set_ylabel('Alpha Cumulative Product')
    axes_schedule[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_schedule


@app.cell
def _(mo):
    mo.md("""## 🔍 Explore Training Data""")
    return


@app.cell
def _(plt, train_loader):
    # Show some training examples
    data_iter_train = iter(train_loader)
    images_train, labels_train = next(data_iter_train)

    fig_train_samples, axes_train_samples = plt.subplots(2, 8, figsize=(16, 4))
    axes_train_samples = axes_train_samples.flatten()

    for i in range(16):
        img_display = images_train[i].squeeze().numpy()
        axes_train_samples[i].imshow(img_display, cmap='gray')
        axes_train_samples[i].set_title(f'Label: {labels_train[i].item()}')
        axes_train_samples[i].axis('off')

    plt.tight_layout()
    plt.suptitle('Sample Training Images', y=1.02, fontsize=14)
    return fig_train_samples


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
    - The model was trained for **10 epochs** on MNIST
    - Uses **MSE loss** for noise prediction
    - **1000 timesteps** for full diffusion process
    """
    )
    return


if __name__ == "__main__":
    app.run()
