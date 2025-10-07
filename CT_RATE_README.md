# CT-RATE Dataset Training Guide

This guide explains how to train the 3D Medical Diffusion model on the CT-RATE dataset.

## Prerequisites

1. **Dataset Format**: Your CT-RATE dataset should contain `.nii.gz` files (NIfTI format)
2. **Data Normalization**: CT images should be normalized to the range `[-1, 1]`
3. **Environment**: Make sure you have the 3DMedDiffusion environment set up

## Dataset Structure

Your CT-RATE dataset should have the following structure:
```
/project/flame/daesungk/CT-RATE/dataset/
├── train/
│   ├── train_1/
│   │   ├── train_1_a/
│   │   │   ├── train_1_a_1.nii.gz
│   │   │   └── train_1_a_2.nii.gz
│   │   └── train_1_b/
│   │       ├── train_1_b_1.nii.gz
│   │       └── train_1_b_2.nii.gz
│   ├── train_2/
│   └── ...
└── val/  (will be created automatically)
    ├── train_5/
    ├── train_12/
    └── ...
```

## Quick Start

### 1. Setup Dataset

Run the setup script to configure your CT-RATE dataset and create a validation split:

```bash
cd 3DMedDiffusion
python setup_CT_RATE.py --dataset-path /project/flame/daesungk/CT-RATE/dataset/train
```

This will:
- Create a validation split (10% of data by default)
- Update the data configuration file
- Create necessary training directories
- Verify your dataset structure

**Optional**: You can customize the validation split:
```bash
python setup_CT_RATE.py --dataset-path /project/flame/daesungk/CT-RATE/dataset/train --validation-split 0.15
```

**Verify only** (without creating validation split):
```bash
python setup_CT_RATE.py --dataset-path /project/flame/daesungk/CT-RATE/dataset/train --verify-only
```

### 2. Verify Configuration

Check that your data configuration file (`config/CT_RATE_data.json`) contains both paths:

```json
{
    "CT_RATE_train": "/project/flame/daesungk/CT-RATE/dataset/train",
    "CT_RATE_val": "/project/flame/daesungk/CT-RATE/dataset/val"
}
```

### 3. Start Training

Begin training with the CT-RATE specific training script:

```bash
python train/train_PatchVolume_CT_RATE.py --config config/PatchVolume_4x_CT_RATE.yaml
```

## Configuration Details

The `PatchVolume_4x_CT_RATE.yaml` configuration is optimized for CT-RATE training:

- **Batch Size**: 16 (adjust based on your GPU memory)
- **Learning Rate**: 3e-4
- **Patch Size**: 64x64x64
- **Downsample**: 4x4x4 compression
- **Training Steps**: 1,000,000 (configurable)
- **Validation Split**: 10% of data (configurable)
- **Training Directory**: `/project/flame/daesungk/3DMedDiffusion_CT_RATE_training`

## Dataset Requirements

### File Structure
- **File Type**: `.nii.gz` (NIfTI compressed)
- **Modality**: CT images
- **Channels**: 1 (grayscale)
- **Normalization**: Values should be in range `[-1, 1]`
- **Organization**: Hierarchical folder structure with patient-level organization

### Data Count
Based on your CSV file, you have approximately **738 training samples** across multiple patients.

## Training Stages

### Stage 1: PatchVolume Autoencoder
- **Purpose**: Learn to compress CT images into latent representations
- **Output**: Checkpoint file for stage 2
- **Dataset**: Uses the new `VQGANDataset_4x_CT_RATE` class

### Stage 2: Fine-tuning (Optional)
After stage 1 completes, you can run stage 2 for refinement:

```bash
python train/train_PatchVolume_stage2.py --config config/PatchVolume_4x_s2_CT_RATE.yaml
```

## Monitoring Training

- **Logs**: Check `/project/flame/daesungk/3DMedDiffusion_CT_RATE_training/logs/` for TensorBoard logs
- **Checkpoints**: Saved in `/project/flame/daesungk/3DMedDiffusion_CT_RATE_training/checkpoints/`
- **Samples**: Generated samples saved in `/project/flame/daesungk/3DMedDiffusion_CT_RATE_training/samples/`

## Key Differences from Original

1. **Custom Dataset Class**: `VQGANDataset_4x_CT_RATE` handles the hierarchical folder structure
2. **Proper Train/Val Split**: Creates a proper validation set instead of using last 40 files
3. **CT-RATE Specific Training**: `train_PatchVolume_CT_RATE.py` uses the custom dataset class
4. **Automatic Validation**: Setup script automatically creates validation split
5. **Project Directory Storage**: All training artifacts stored in `/project/flame/daesungk/`

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size in the config file
2. **No .nii.gz files found**: Ensure your dataset contains NIfTI files
3. **Data loading errors**: Check file permissions and paths
4. **Validation split issues**: Verify the setup script completed successfully

### Performance Tips

- Use SSD storage for faster data loading
- Adjust `num_workers` based on your system
- Monitor GPU memory usage during training
- The hierarchical structure allows for efficient data organization

## Next Steps

After training the autoencoder:

1. **Generate Latents**: Use the trained model to encode images to latent space
2. **Train Diffusion Model**: Train the BiFlowNet on the generated latents
3. **Inference**: Generate new CT images using the trained model

## Support

For issues specific to CT-RATE training, check:
- Dataset format and normalization
- Configuration file paths
- GPU memory requirements
- File permissions and access
- Validation split creation
