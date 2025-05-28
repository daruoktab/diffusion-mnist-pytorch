# MNIST Diffusion Model - Learning Project

A PyTorch implementation of a diffusion model for generating MNIST handwritten digits using the DeepInv library. This project is currently in development and serves as a learning exercise for understanding diffusion models.

## 🚧 Project Status

**Work in Progress** - This project is actively being developed and is not yet complete.

### Current Implementation
- ✅ Basic U-Net architecture using DeepInv's DiffUNet
- ✅ MNIST data loading and preprocessing
- ✅ Diffusion process setup (forward noising)
- ✅ Training loop implementation
- ✅ Model saving functionality
- ❌ Image generation from trained model (in progress)
- ❌ Sampling and denoising process
- ❌ Result visualization and evaluation

## 📁 Project Structure

```
diffusion-mnist-pytorch/
├── belajardiffusion.ipynb    # Jupyter notebook (main development)
├── trainingdiffusion.py      # Marimo app version (unfinished)
├── README.md                 # This file
└── data/                     # MNIST dataset (auto-downloaded)
```

## 🔧 Requirements

```bash
pip install torch torchvision deepinv marimo matplotlib numpy
```

### Key Dependencies
- **PyTorch**: Deep learning framework
- **DeepInv**: Computer vision library with pre-built diffusion models
- **Marimo**: Interactive notebook environment (for .py version)
- **Torchvision**: For MNIST dataset and transforms

## 🚀 Current Usage

### Jupyter Notebook (Primary)
Open and run the notebook for interactive development:
```bash
jupyter notebook belajardiffusion.ipynb
```

### Marimo App (Alternative)
Run the Marimo version (incomplete):
```bash
marimo run trainingdiffusion.py
```

## 🏗️ Current Architecture

### Model Configuration
- **Architecture**: DiffUNet from DeepInv library
- **Input/Output**: 1 channel (grayscale)
- **Image Size**: 32x32 (resized from 28x28)
- **Batch Size**: 48 (notebook) / 64 (marimo)

### Diffusion Parameters
- **Timesteps**: 1000
- **Beta Schedule**: Linear from 1e-4 to 0.02
- **Loss Function**: MSE between predicted and actual noise

### Training Setup
- **Optimizer**: Adam (lr=1e-4)
- **Epochs**: 5 (notebook) / 10 (marimo planned)
- **Device**: CUDA if available, else CPU

## 📊 Implementation Details

### Forward Diffusion Process
The model adds Gaussian noise to MNIST images over 1000 timesteps:
```python
noisy_imgs = (
    sqrt_alphas_cumprod[t, None, None, None] * imgs +
    sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
)
```

### Training Objective
The model learns to predict the noise added at each timestep:
- Sample random timestep `t`
- Add corresponding noise level to clean image
- Train U-Net to predict the added noise
- Minimize MSE loss between predicted and actual noise

## 🎯 Next Steps (TODO)

### Immediate Goals
1. **Complete the reverse diffusion process** for image generation
2. **Implement sampling algorithm** to generate new digits
3. **Add result visualization** to see generated samples
4. **Finish the Marimo app version** with complete functionality

### Future Enhancements
- [ ] Add classifier-free guidance for conditional generation
- [ ] Implement different noise schedules (cosine, etc.)
- [ ] Add FID/IS metrics for evaluation
- [ ] Experiment with different U-Net architectures
- [ ] Add interpolation between digits

## 📚 Learning Resources

This project is based on understanding:
- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [DeepInv Documentation](https://deepinv.github.io/)
- Diffusion model fundamentals and implementation

## 🔍 Current Limitations

- **No generation capability yet** - can only train the model
- **Missing reverse process** - need to implement sampling
- **No evaluation metrics** - need to add quality assessment
- **Marimo version incomplete** - training code is commented out

## 💡 Development Notes

### Why DeepInv?
Using DeepInv's DiffUNet provides:
- Pre-built, tested U-Net architecture
- Proper time embedding handling
- Simplified model setup for learning purposes

### Learning Focus
This project emphasizes understanding:
- Forward and reverse diffusion processes
- Noise prediction training paradigm
- U-Net architecture for diffusion
- PyTorch implementation details

## 🤝 Contributing

This is a personal learning project, but suggestions and improvements are welcome!

## 📄 License

Educational/Learning project - feel free to use and modify.
