#!/usr/bin/env python3
"""
Setup script for CT-RATE dataset training
This script helps prepare the CT-RATE dataset for 3D Medical Diffusion training.
"""

import os
import json
import argparse
import random
import shutil
from pathlib import Path
import pandas as pd

def setup_ct_rate_dataset(dataset_path, output_config_path, validation_split=0.1):
    """
    Setup CT-RATE dataset configuration with validation split
    
    Args:
        dataset_path: Path to your CT-RATE dataset directory
        output_config_path: Path to save the updated data configuration
        validation_split: Fraction of data to use for validation (default: 0.1 = 10%)
    """
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist!")
        return False
    
    # Create validation directory
    base_dir = Path(dataset_path).parent
    val_dir = base_dir / "val"
    val_dir.mkdir(exist_ok=True)
    
    # Get all training folders
    train_folders = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and item.startswith('train_'):
            train_folders.append(item)
    
    if not train_folders:
        print(f"Error: No training folders found in {dataset_path}")
        return False
    
    print(f"Found {len(train_folders)} training folders")
    
    # Randomly select folders for validation
    random.seed(42)  # For reproducible splits
    num_val = max(1, int(len(train_folders) * validation_split))
    val_folders = random.sample(train_folders, num_val)
    
    print(f"Creating validation split with {num_val} folders ({validation_split*100:.1f}%)")
    
    # Move validation folders
    for folder in val_folders:
        src = os.path.join(dataset_path, folder)
        dst = os.path.join(val_dir, folder)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
        print(f"Moved {folder} to validation set")
    
    # Count files in both sets
    train_files = []
    val_files = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.nii.gz'):
                train_files.append(os.path.join(root, file))
    
    for root, dirs, files in os.walk(val_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                val_files.append(os.path.join(root, file))
    
    print(f"Training set: {len(train_files)} .nii.gz files")
    print(f"Validation set: {len(val_files)} .nii.gz files")
    
    # Create data configuration with both paths
    data_config = {
        "CT_RATE_train": str(dataset_path),
        "CT_RATE_val": str(val_dir)
    }
    
    # Save configuration
    with open(output_config_path, 'w') as f:
        json.dump(data_config, f, indent=4)
    
    print(f"Data configuration saved to {output_config_path}")
    print(f"Training path: {dataset_path}")
    print(f"Validation path: {val_dir}")
    
    return True

def create_training_directories():
    """Create necessary training directories"""
    base_dir = Path("/project/flame/daesungk/3DMedDiffusion_CT_RATE_training")
    
    # Create directories
    (base_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base_dir / "logs").mkdir(parents=True, exist_ok=True)
    (base_dir / "samples").mkdir(parents=True, exist_ok=True)
    
    print(f"Created training directories in {base_dir}")

def verify_dataset_structure(dataset_path):
    """Verify the dataset structure and count files"""
    print("\nVerifying dataset structure...")
    
    total_files = 0
    folder_counts = {}
    
    for root, dirs, files in os.walk(dataset_path):
        nii_files = [f for f in files if f.endswith('.nii.gz')]
        if nii_files:
            folder_counts[root] = len(nii_files)
            total_files += len(nii_files)
    
    print(f"Total .nii.gz files found: {total_files}")
    print("\nFolder structure:")
    for folder, count in sorted(folder_counts.items()):
        rel_path = os.path.relpath(folder, dataset_path)
        print(f"  {rel_path}: {count} files")
    
    return total_files

def main():
    parser = argparse.ArgumentParser(description="Setup CT-RATE dataset for 3D Medical Diffusion training")
    parser.add_argument("--dataset-path", type=str, 
                       default="/project/flame/daesungk/CT-RATE/dataset/train",
                       help="Path to your CT-RATE dataset directory")
    parser.add_argument("--output-config", type=str, 
                       default="config/CT_RATE_data.json",
                       help="Output path for data configuration file")
    parser.add_argument("--validation-split", type=float, default=0.1,
                       help="Fraction of data to use for validation (default: 0.1)")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify dataset structure without creating validation split")
    
    args = parser.parse_args()
    
    print("Setting up CT-RATE dataset for 3D Medical Diffusion training...")
    print("=" * 60)
    
    if args.verify_only:
        verify_dataset_structure(args.dataset_path)
        return
    
    # Setup dataset
    if setup_ct_rate_dataset(args.dataset_path, args.output_config, args.validation_split):
        # Create training directories
        create_training_directories()
        
        # Verify final structure
        print("\nFinal dataset structure:")
        verify_dataset_structure(args.dataset_path)
        
        print("\n" + "=" * 60)
        print("Setup completed successfully!")
        print("\nNext steps:")
        print("1. Verify the validation split looks correct")
        print("2. Check the updated data configuration in config/CT_RATE_data.json")
        print("3. Run training with:")
        print(f"   python train/train_PatchVolume_CT_RATE.py --config config/PatchVolume_4x_CT_RATE.yaml")
        print("\nNote: Make sure your CT images are normalized to [-1, 1] range")
    else:
        print("Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
