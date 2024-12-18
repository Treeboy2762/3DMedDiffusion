# 3D MedDiffusion: A 3D Medical Diffusion Model for Controllable and High-quality Medical Image Generation
[![Static Badge](https://img.shields.io/badge/Project-page-blue)](https://shanghaitech-impact.github.io/3D-MedDiffusion.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2402.19043-b31b1b.svg)](https://arxiv.org/abs/2412.13059)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20Page-orange)](https://huggingface.co/MMorss/3D-MedDiffusion)

This is the official PyTorch implementation of the paper **3D MedDiffusion: A 3D Medical Diffusion Model for Controllable and High-quality Medical Image Generation** 

![SMD in Action](assets/gif_github.gif)


## Paper Abstract
The generation of medical images presents significant challenges due to their high-resolution and three-dimensional nature. Existing methods often yield suboptimal performance in generating high-quality 3D medical images, and there is currently no universal generative framework for medical imaging.
In this paper, we introduce the 3D Medical Diffusion (3D MedDiffusion) model for controllable, high-quality 3D medical image generation. 
3D MedDiffusion incorporates a novel, highly efficient Patch-Volume Autoencoder that compresses medical images into latent space through patch-wise encoding and recovers back into image space through volume-wise decoding.
Additionally, we design a new noise estimator to capture both local details and global structure information during diffusion denoising process.
3D MedDiffusion can generate fine-detailed, high-resolution images (up to 512x512x512) and effectively adapt to various downstream tasks as it is trained on large-scale datasets covering CT and MRI modalities and different anatomical regions (from head to leg).
Experimental results demonstrate that 3D MedDiffusion surpasses state-of-the-art methods in generative quality and exhibits strong generalizability across tasks such as sparse-view CT reconstruction, fast MRI reconstruction, and data augmentation.
