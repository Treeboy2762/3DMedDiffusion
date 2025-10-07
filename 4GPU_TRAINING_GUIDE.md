# 4-GPU Distributed Training Guide for CT-RATE

This guide explains how to train the 3D Medical Diffusion model on the CT-RATE dataset using 4 GPUs with Slurm.

## 🚀 Quick Start

### 1. Submit the Training Job
```bash
cd 3DMedDiffusion
chmod +x submit_training.sh
./submit_training.sh
```

### 2. Monitor Your Job
```bash
# Check job status
squeue -u $USER

# Monitor output
tail -f slurm-<JOB_ID>.out

# Check GPU usage
srun --jobid=<JOB_ID> nvidia-smi
```

## 📁 Files Created for 4-GPU Training

1. **`train_CT_RATE_4GPU.slurm`** - Main Slurm job script
2. **`train/train_PatchVolume_CT_RATE_distributed.py`** - Distributed training script
3. **`config/PatchVolume_4x_CT_RATE_4GPU.yaml`** - 4-GPU optimized configuration
4. **`submit_training.sh`** - Easy submission script

## ⚙️ Configuration Details

### 4-GPU Specific Settings
- **Total Batch Size**: 64 (16 per GPU)
- **Total Workers**: 16 (4 per GPU)
- **GPUs**: 4
- **Strategy**: DDP (Distributed Data Parallel)
- **Backend**: gloo
- **Sync BatchNorm**: Enabled for stability
- **Gradient Clipping**: Enabled

### Key Optimizations
- **Effective Batch Size**: Automatically calculated per GPU
- **Worker Distribution**: Balanced across GPUs
- **Memory Management**: Pin memory enabled
- **Stability**: Gradient clipping and sync batch normalization

## 🔧 Slurm Job Configuration

```bash
#SBATCH -J CT_RATE_training
#SBATCH -p preempt            # Queue name
#SBATCH -t 72:00:00          # Time limit (3 days)
#SBATCH --gpus=4             # 4 GPUs
#SBATCH -N 1                 # 1 node
#SBATCH --account=aperer1    # Your account
```

## 📊 Training Performance

### Expected Benefits
- **~4x Speedup**: Compared to single GPU training
- **Larger Effective Batch Size**: Better gradient estimates
- **Faster Convergence**: Due to increased batch size
- **Better Memory Utilization**: Distributed across GPUs

### Memory Requirements
- **Per GPU**: ~8-12 GB (depending on batch size)
- **Total**: ~32-48 GB across all GPUs
- **System Memory**: ~64 GB recommended

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   ```bash
   # Reduce batch size in config
   batch_size: 48  # Instead of 64
   ```

2. **Job Hanging**
   ```bash
   # Check if all GPUs are accessible
   srun --jobid=<JOB_ID> nvidia-smi
   ```

3. **Communication Errors**
   ```bash
   # Check network connectivity between GPUs
   # Ensure gloo backend is working
   ```

### Debug Commands
```bash
# Check job logs
cat slurm-<JOB_ID>.out
cat slurm-<JOB_ID>.err

# Check GPU status
srun --jobid=<JOB_ID> nvidia-smi

# Check job details
scontrol show job <JOB_ID>
```

## 📈 Monitoring Training

### TensorBoard Logs
```bash
# Logs are saved to:
/project/flame/daesungk/3DMedDiffusion_CT_RATE_training/logs/

# Monitor remotely (if accessible):
tensorboard --logdir=/project/flame/daesungk/3DMedDiffusion_CT_RATE_training/logs/
```

### Checkpoints
```bash
# Checkpoints saved to:
/project/flame/daesungk/3DMedDiffusion_CT_RATE_training/checkpoints/

# Resume from checkpoint:
# Edit config file to add resume_from_checkpoint path
```

## 🔄 Resuming Training

### From Checkpoint
1. Edit `config/PatchVolume_4x_CT_RATE_4GPU.yaml`
2. Add checkpoint path:
   ```yaml
   model:
     resume_from_checkpoint: /project/flame/daesungk/3DMedDiffusion_CT_RATE_training/checkpoints/latest_checkpoint.ckpt
   ```
3. Resubmit the job

### Job Restart
```bash
# If job fails, simply resubmit:
./submit_training.sh
```

## 📋 Job Management Commands

```bash
# Submit job
sbatch train_CT_RATE_4GPU.slurm

# Check job status
squeue -u $USER

# Cancel job
scancel <JOB_ID>

# Hold job (pause)
scontrol hold <JOB_ID>

# Release held job
scontrol release <JOB_ID>

# Check job efficiency
seff <JOB_ID>
```

## 🎯 Best Practices

1. **Monitor GPU Utilization**: Use `nvidia-smi` to ensure all GPUs are working
2. **Check Logs Regularly**: Monitor for errors or warnings
3. **Save Checkpoints**: Enable frequent checkpointing for recovery
4. **Memory Management**: Start with smaller batch size if OOM occurs
5. **Network Stability**: Ensure stable network for GPU communication

## 🆘 Getting Help

If you encounter issues:

1. **Check Slurm logs**: `cat slurm-<JOB_ID>.out`
2. **Verify GPU access**: `nvidia-smi`
3. **Check configuration**: Ensure all paths are correct
4. **Monitor resources**: Check if queue has enough resources

## 📝 Notes

- **Environment**: Make sure `meddiff` conda environment is activated
- **Dependencies**: All required packages should be in the conda environment
- **Storage**: Training artifacts saved to `/project/flame/daesungk/`
- **Time Limit**: Adjust `-t` parameter based on your needs
- **Queue**: Use appropriate queue for your account (`preempt`, `gpu`, etc.)
