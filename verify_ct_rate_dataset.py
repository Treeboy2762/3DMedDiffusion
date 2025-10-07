#!/usr/bin/env python3
"""
Verification script for CT-RATE dataset
This script checks the dataset structure before running the main setup.
"""

import os
import glob
from pathlib import Path

def verify_ct_rate_dataset(dataset_path):
    """
    Verify the CT-RATE dataset structure
    
    Args:
        dataset_path: Path to the CT-RATE dataset
    """
    
    print("=" * 60)
    print("CT-RATE Dataset Verification")
    print("=" * 60)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path {dataset_path} does not exist!")
        return False
    
    print(f"✅ Dataset path exists: {dataset_path}")
    
    # Check for training folders
    train_folders = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and item.startswith('train_'):
            train_folders.append(item)
    
    if not train_folders:
        print("❌ Error: No training folders found!")
        return False
    
    print(f"✅ Found {len(train_folders)} training folders")
    
    # Check structure of first few folders
    print("\n📁 Checking folder structure...")
    total_files = 0
    folder_details = {}
    
    for folder in sorted(train_folders)[:5]:  # Check first 5 folders
        folder_path = os.path.join(dataset_path, folder)
        subfolders = []
        files = []
        
        for subitem in os.listdir(folder_path):
            subitem_path = os.path.join(folder_path, subitem)
            if os.path.isdir(subitem_path):
                subfolders.append(subitem)
                # Count .nii.gz files in subfolder
                nii_files = glob.glob(os.path.join(subitem_path, "*.nii.gz"))
                files.extend(nii_files)
        
        folder_details[folder] = {
            'subfolders': len(subfolders),
            'files': len(files)
        }
        total_files += len(files)
        
        print(f"  {folder}: {len(subfolders)} subfolders, {len(files)} .nii.gz files")
    
    if len(train_folders) > 5:
        print(f"  ... and {len(train_folders) - 5} more folders")
    
    # Count total files
    all_files = glob.glob(os.path.join(dataset_path, "**/*.nii.gz"), recursive=True)
    total_files = len(all_files)
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Total training folders: {len(train_folders)}")
    print(f"  Total .nii.gz files: {total_files}")
    
    # Check file sizes
    if all_files:
        sizes = [os.path.getsize(f) for f in all_files[:10]]  # Check first 10 files
        avg_size_mb = sum(sizes) / len(sizes) / (1024 * 1024)
        print(f"  Average file size: {avg_size_mb:.1f} MB")
        
        # Check if files are readable
        print("\n🔍 Testing file readability...")
        test_file = all_files[0]
        try:
            import nibabel as nib
            img = nib.load(test_file)
            print(f"  ✅ Successfully loaded: {os.path.basename(test_file)}")
            print(f"     Shape: {img.shape}")
            print(f"     Data type: {img.get_data_dtype()}")
        except ImportError:
            print("  ⚠️  nibabel not available - cannot test file loading")
        except Exception as e:
            print(f"  ❌ Error loading file: {e}")
    
    print("\n" + "=" * 60)
    print("Verification Complete!")
    print("=" * 60)
    
    if total_files > 0:
        print("✅ Dataset appears to be ready for training!")
        print(f"\nNext step: Run the setup script:")
        print(f"python setup_CT_RATE.py --dataset-path {dataset_path}")
    else:
        print("❌ No .nii.gz files found. Please check your dataset.")
    
    return total_files > 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify CT-RATE dataset structure")
    parser.add_argument("--dataset-path", type=str, 
                       default="/project/flame/daesungk/CT-RATE/dataset/train",
                       help="Path to CT-RATE dataset")
    
    args = parser.parse_args()
    
    verify_ct_rate_dataset(args.dataset_path)

if __name__ == "__main__":
    main()
