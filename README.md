# 3D MedDiffusion: A 3D Medical Diffusion Model for Controllable and High-quality Medical Image Generation
[![Static Badge](https://img.shields.io/badge/Project-page-blue)](https://shanghaitech-impact.github.io/3D-MedDiffusion/)
[![arXiv](https://img.shields.io/badge/arXiv-2402.19043-b31b1b.svg)](https://arxiv.org/abs/2412.13059)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20Page-orange)](https://huggingface.co/MMorss/3D-MedDiffusion)

Adaptation of the paper **3D MedDiffusion: A 3D Medical Diffusion Model for Controllable and High-quality Medical Image Generation** for CT-RATE dataset

![SMD in Action](assets/gif_github.gif)

## Training 
### PatchVolume Autoencoder — Stage 1


```
sbatch train_CT_RATE_stage1.slurm
```
**Note:**  
1. All training images should be normalized to `[-1, 1]`.  
2. Update the `default_root_dir`and `root_dir` fileds in `config/PatchVolume_4x.yaml` / `config/PatchVolume_8x.yaml` to match your local paths.
3. Provide a `data.json` following the format shown in the `config/PatchVolume_data.json` example.


### PatchVolume Autoencoder — Stage 2

```
sbatch train_CT_RATE_stage1.slurm
```

**Note:** Set the `resume_from_checkpoint` in `PatchVolume_4x.yaml` / `PatchVolume_8x.yaml` to the checkpoint path from Stage 1 training.


### Encode the Images to latents 
```
sbatch generate_latents.slurm
```

### BiFlowNet
```
sbatch train_CT_RATE_latent.slurm
```

## Inference
```
sbatch test_generation.slurm
```
**Note:**  Make sure your GPU has at least 40 GB of memory available to run inference at all supported resolutions.
