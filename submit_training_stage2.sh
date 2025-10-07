#!/bin/bash
# Simple script to submit the CT-RATE training job (8 GPUs)

echo "Submitting CT-RATE training job to Slurm (2 GPUs, simple multi-GPU)..."

# Submit the job
job_id=$(sbatch train_CT_RATE_stage2.slurm | awk '{print $4}')

echo "Job submitted successfully!"
echo "Job ID: $job_id"
echo ""
echo "To monitor your job:"
echo "  squeue -j $job_id"
echo ""
echo "To check the output:"
echo "  tail -f slurm-${job_id}.out"
echo ""
echo "To cancel the job:"
echo "  scancel $job_id"
echo ""
echo "To check GPU usage:"
echo "  srun --jobid=$job_id nvidia-smi"
echo ""
echo "Note: This job uses PyTorch Lightning's built-in multi-GPU support"
echo "      No distributed multiprocessing - simpler and more reliable!"
echo "      Using 8 GPUs for maximum training speed!"
