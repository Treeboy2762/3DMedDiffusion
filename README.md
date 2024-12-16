# SMD: A Stable Medical Diffusion Model for Controllable and High-quality 3D Medical Images Generation
[![Static Badge](https://img.shields.io/badge/Project-page-blue)](..)
[![arXiv](https://img.shields.io/badge/arXiv-2402.19043-b31b1b.svg)](..)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20Page-orange)](..)

This is the official PyTorch implementation of the paper **SMD: A Stable Medical Diffusion Model for Controllable and High-quality 3D Medical Images Generation** 




## Paper Abstract
The generative modeling of medical images presents significant challenges due to their high-resolution and three-dimensional nature. Existing methods often yield suboptimal performance in generating high-quality 3D medical images, and there is currently no universal generative framework for medical imaging. In this paper, we introduce the Stable Medical Diffusion (SMD) model for controllable, high-quality 3D medical image generation. SMD incorporates a novel, highly efficient Patch-Volume Autoencoder that compresses medical images into latent space through patch-wise encoding and recovers back into image space through volume-wise decoding. Additionally, we design a new noise estimator to capture both local details and global structure information during diffusion denoising process. SMD can generate fine-detailed, high-resolution images (up to 512x512x512) and effectively adapt to various downstream tasks as it is trained on large-scale medical datasets covering multiple modalities (CT and MRI) and anatomical regions (from head to leg).
Experimental results demonstrate that SMD surpasses state-of-the-art methods in generative quality and exhibits strong generalizability across tasks such as sparse-view CT reconstruction, fast MRI reconstruction, and data augmentation.

