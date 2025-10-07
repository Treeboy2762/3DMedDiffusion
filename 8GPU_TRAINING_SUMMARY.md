# 8-GPU Training Setup for CT-RATE

## 🚀 Quick Start

```bash
cd 3DMedDiffusion
chmod +x submit_training_8GPU.sh
./submit_training_8GPU.sh
```

## 📁 Files for 8-GPU Training

1. **`train_CT_RATE_8GPU.slurm`** - Slurm job script requesting 8 GPUs
2. **`config/PatchVolume_4x_CT_RATE_8GPU.yaml`** - 8-GPU optimized configuration
3. **`submit_training_8GPU.sh`** - Easy submission script
4. **`train/train_PatchVolume_CT_RATE_simple.py`** - Multi-GPU training script

## ⚙️ 8-GPU Configuration

- **Total GPUs**: 8
- **Total Batch Size**: 128 (16 per GPU)
- **Total Workers**: 32 (4 per GPU)
- **Strategy**: Auto (PyTorch Lightning chooses best)
- **Storage**: `/project/flame/daesungk/3DMedDiffusion_CT_RATE_training/`

## 📊 Performance Benefits

- **~8x Speedup** compared to single GPU
- **Larger effective batch size** for better gradients
- **Faster convergence** due to increased batch size
- **Better memory utilization** across all 8 GPUs

## 🔧 Key Features

- **Simple approach**: No distributed multiprocessing complexity
- **PyTorch Lightning handles multi-GPU**: Automatic strategy selection
- **Automatic batch distribution**: 128 total → 16 per GPU
- **Sync batch normalization**: Ensures training stability
- **Gradient clipping**: Prevents gradient explosion

## 📈 Monitoring

- **Logs**: `/project/flame/daesungk/3DMedDiffusion_CT_RATE_training/logs/`
- **Checkpoints**: `/project/flame/daesungk/3DMedDiffusion_CT_RATE_training/checkpoints/`
- **TensorBoard**: `CT_RATE_model_8GPU` experiment name

## 🎯 Expected Results

With 8 GPUs, you should see:
- **Significantly faster training** per epoch
- **Better gradient estimates** due to larger batch size
- **More stable training** with proper batch normalization
- **Efficient GPU utilization** across all devices

## 📝 Notes

- **Memory per GPU**: ~6-8 GB (depending on batch size)
- **Total memory**: ~48-64 GB across all GPUs
- **System memory**: ~128 GB recommended
- **Queue**: Uses `preempt` queue (adjust if needed)
- **Time limit**: 48 hours (adjustable in Slurm script)
