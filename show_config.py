#!/usr/bin/env python3
"""
Show current CT-RATE training configuration
"""

import os
import json
from pathlib import Path

def show_config():
    """Display the current CT-RATE training configuration"""
    
    print("=" * 60)
    print("CT-RATE Training Configuration")
    print("=" * 60)
    
    # Check data configuration
    data_config_path = "config/CT_RATE_data.json"
    if os.path.exists(data_config_path):
        with open(data_config_path, 'r') as f:
            data_config = json.load(f)
        
        print("📁 Dataset Configuration:")
        for key, value in data_config.items():
            print(f"  {key}: {value}")
            
        # Check if paths exist
        print("\n🔍 Path Verification:")
        for key, path in data_config.items():
            if os.path.exists(path):
                nii_files = len(list(Path(path).rglob("*.nii.gz")))
                print(f"  ✅ {key}: {path} ({nii_files} .nii.gz files)")
            else:
                print(f"  ❌ {key}: {path} (NOT FOUND)")
    else:
        print("❌ Data configuration file not found: config/CT_RATE_data.json")
    
    # Check training configuration
    train_config_path = "config/PatchVolume_4x_CT_RATE.yaml"
    if os.path.exists(train_config_path):
        print(f"\n⚙️  Training Configuration: {train_config_path}")
        print("  ✅ Found")
    else:
        print(f"\n❌ Training configuration not found: {train_config_path}")
    
    # Check training directory
    train_dir = "/project/flame/daesungk/3DMedDiffusion_CT_RATE_training"
    if os.path.exists(train_dir):
        print(f"\n📂 Training Directory: {train_dir}")
        print("  ✅ Exists")
        
        # Check subdirectories
        subdirs = ["checkpoints", "logs", "samples"]
        for subdir in subdirs:
            subdir_path = os.path.join(train_dir, subdir)
            if os.path.exists(subdir_path):
                print(f"    ✅ {subdir}/")
            else:
                print(f"    ❌ {subdir}/ (missing)")
    else:
        print(f"\n❌ Training directory not found: {train_dir}")
    
    # Check dataset class
    dataset_class_path = "dataset/vqgan_4x_ct_rate.py"
    if os.path.exists(dataset_class_path):
        print(f"\n🐍 CT-RATE Dataset Class: {dataset_class_path}")
        print("  ✅ Found")
    else:
        print(f"\n❌ CT-RATE dataset class not found: {dataset_class_path}")
    
    # Check training script
    train_script_path = "train/train_PatchVolume_CT_RATE.py"
    if os.path.exists(train_script_path):
        print(f"\n🚀 CT-RATE Training Script: {train_script_path}")
        print("  ✅ Found")
    else:
        print(f"\n❌ CT-RATE training script not found: {train_script_path}")
    
    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    
    if os.path.exists(data_config_path) and os.path.exists(train_config_path):
        print("✅ Configuration appears complete!")
        print("\nNext steps:")
        print("1. Run setup script: python setup_CT_RATE.py")
        print("2. Start training: python train/train_PatchVolume_CT_RATE.py --config config/PatchVolume_4x_CT_RATE.yaml")
    else:
        print("❌ Configuration incomplete. Please check missing files above.")
    
    print("\nNote: All training artifacts will be saved to:")
    print(f"   {train_dir}")

if __name__ == "__main__":
    show_config()
