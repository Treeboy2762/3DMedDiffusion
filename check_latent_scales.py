#!/usr/bin/env python3
"""
Check the distribution and scales of latent variables in training data
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "."))
sys.path.append(project_root)

import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from tqdm import tqdm

def load_latent_files(latent_dir):
    """Load all .npy files from the latent directory"""
    latent_files = glob.glob(os.path.join(latent_dir, "*.npy"))
    
    if not latent_files:
        print(f"❌ No .npy files found in {latent_dir}")
        return []
    
    print(f"Found {len(latent_files)} latent files")
    return sorted(latent_files)

def analyze_latent_distribution(latent_files):
    """Analyze the distribution of latent variables - memory efficient"""
    print(f"\n🔍 ANALYZING LATENT DISTRIBUTIONS")
    print("=" * 50)
    
    all_stats = []
    all_mins = []
    all_maxs = []
    all_means = []
    
    print(f"Analyzing {len(latent_files)} files (memory efficient)")
    
    for i, file_path in enumerate(tqdm(latent_files, desc="Loading latents")):
        try:
            # Load latent data
            latent = np.load(file_path)
            
            # Basic statistics - only store what we need
            stats = {
                'file': os.path.basename(file_path),
                'shape': latent.shape,
                'min': float(latent.min()),
                'max': float(latent.max()),
                'mean': float(latent.mean()),
                'std': float(latent.std()),
                'range': float(latent.max() - latent.min())
            }
            
            all_stats.append(stats)
            all_mins.append(stats['min'])
            all_maxs.append(stats['max'])
            all_means.append(stats['mean'])
            
            # Print individual file stats for first 5 files
            if i < 5:
                print(f"\n📁 {stats['file']}")
                print(f"   Shape: {stats['shape']}")
                print(f"   Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"   Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            
            # Clear memory immediately
            del latent
                
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    return all_stats, all_mins, all_maxs, all_means

def check_normalization(all_stats, all_mins, all_maxs, all_means):
    """Check if latents are properly normalized to [-1, 1]"""
    print(f"\n🎯 NORMALIZATION CHECK")
    print("=" * 50)
    
    # Overall statistics from stored mins/maxs
    if all_mins and all_maxs:
        overall_min = min(all_mins)
        overall_max = max(all_maxs)
        overall_mean = np.mean(all_means)
        
        print(f"📊 OVERALL STATISTICS (all latents combined):")
        print(f"   Range: [{overall_min:.4f}, {overall_max:.4f}]")
        print(f"   Mean: {overall_mean:.4f}")
        print(f"   Files analyzed: {len(all_stats)}")
        
        # Check if within [-1, 1] range
        within_range = (overall_min >= -1.0) and (overall_max <= 1.0)
        print(f"\n✅ Within [-1, 1] range: {within_range}")
        
        if not within_range:
            print(f"❌ ISSUE FOUND: Latents are NOT properly normalized!")
            print(f"   Min value: {overall_min:.4f} (should be >= -1.0)")
            print(f"   Max value: {overall_max:.4f} (should be <= 1.0)")
            print(f"   Range span: {overall_max - overall_min:.4f}")
        else:
            print(f"✅ Latents are properly normalized to [-1, 1]")
        
        # Check individual file ranges
        print(f"\n📋 INDIVIDUAL FILE RANGES:")
        out_of_range_files = []
        for stats in all_stats:
            in_range = (stats['min'] >= -1.0) and (stats['max'] <= 1.0)
            status = "✅" if in_range else "❌"
            print(f"   {status} {stats['file']}: [{stats['min']:.4f}, {stats['max']:.4f}]")
            
            if not in_range:
                out_of_range_files.append(stats['file'])
        
        if out_of_range_files:
            print(f"\n⚠️  {len(out_of_range_files)} files are out of [-1, 1] range")
        else:
            print(f"\n✅ All files are within [-1, 1] range")
    
    return overall_min, overall_max, overall_mean, 0.0  # std not calculated

def create_visualizations(all_stats, output_dir):
    """Create simple visualization plots of the latent distributions"""
    print(f"\n📈 CREATING VISUALIZATIONS")
    print("=" * 50)
    
    if not all_stats:
        print("No data to visualize")
        return
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ matplotlib not available, skipping visualizations")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract mins and maxs for plotting
    mins = [stats['min'] for stats in all_stats]
    maxs = [stats['max'] for stats in all_stats]
    means = [stats['mean'] for stats in all_stats]
    
    # Create simple plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Latent Variable Range Analysis', fontsize=16)
    
    # 1. Min values distribution
    axes[0, 0].hist(mins, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(-1, color='red', linestyle='--', label='Target: -1')
    axes[0, 0].set_title('Distribution of Min Values')
    axes[0, 0].set_xlabel('Min Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Max values distribution
    axes[0, 1].hist(maxs, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(1, color='red', linestyle='--', label='Target: 1')
    axes[0, 1].set_title('Distribution of Max Values')
    axes[0, 1].set_xlabel('Max Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Mean values distribution
    axes[1, 0].hist(means, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', label='Target: 0')
    axes[1, 0].set_title('Distribution of Mean Values')
    axes[1, 0].set_xlabel('Mean Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Range vs Mean scatter
    ranges = [stats['max'] - stats['min'] for stats in all_stats]
    axes[1, 1].scatter(means, ranges, alpha=0.6, color='purple')
    axes[1, 1].axhline(2, color='red', linestyle='--', label='Target Range: 2')
    axes[1, 1].set_title('Range vs Mean')
    axes[1, 1].set_xlabel('Mean Value')
    axes[1, 1].set_ylabel('Range (Max - Min)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'latent_range_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved visualization: {plot_path}")
    
    plt.close()

def suggest_fixes(overall_min, overall_max, overall_mean, overall_std):
    """Suggest fixes if normalization issues are found"""
    print(f"\n🛠️  SUGGESTED FIXES")
    print("=" * 50)
    
    if overall_min < -1.0 or overall_max > 1.0:
        print("❌ NORMALIZATION ISSUE DETECTED!")
        print("\n🔧 Potential fixes:")
        
        # Calculate scaling factors
        current_range = overall_max - overall_min
        target_range = 2.0  # [-1, 1] has range of 2
        
        print(f"1. **Rescale latents to [-1, 1] range:**")
        print(f"   Current range: [{overall_min:.4f}, {overall_max:.4f}] (span: {current_range:.4f})")
        print(f"   Target range: [-1.0, 1.0] (span: 2.0)")
        print(f"   Scale factor: {target_range / current_range:.4f}")
        
        print(f"\n2. **Update generation script scaling:**")
        print(f"   Instead of: samples = (samples + 1.0) / 2.0")
        print(f"   Use: samples = (samples - {overall_min:.4f}) / {current_range:.4f} * 2.0 - 1.0")
        
        print(f"\n3. **Check training data preprocessing:**")
        print(f"   Ensure latents are normalized during training")
        print(f"   Verify autoencoder output is in correct range")
        
        print(f"\n4. **Alternative: Update autoencoder codebook range:**")
        print(f"   Modify codebook to match actual latent range")
        print(f"   Update: [{overall_min:.4f}, {overall_max:.4f}]")
        
    else:
        print("✅ Latents are properly normalized!")
        print("The noise issue is likely due to other factors:")
        print("1. Insufficient DDIM steps (try 100-200)")
        print("2. Missing guidance scale")
        print("3. Suboptimal checkpoint selection")
        print("4. Text conditioning issues")

def main():
    print("🔍 LATENT VARIABLE SCALE ANALYSIS")
    print("=" * 50)
    
    # Set paths
    latent_dir = "/tmp/gcsfuse_CTRATE/train_latents"
    output_dir = "./latent_analysis_results"
    
    print(f"Latent directory: {latent_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if latent directory exists
    if not os.path.exists(latent_dir):
        print(f"❌ Latent directory not found: {latent_dir}")
        print("Please check the path and ensure the directory exists")
        return
    
    # Load latent files
    latent_files = load_latent_files(latent_dir)
    if not latent_files:
        return
    
    # Analyze distributions
    all_stats, all_mins, all_maxs, all_means = analyze_latent_distribution(latent_files)
    
    if not all_stats:
        print("❌ No valid latent files found")
        return
    
    # Check normalization
    overall_min, overall_max, overall_mean, overall_std = check_normalization(all_stats, all_mins, all_maxs, all_means)
    
    # Create visualizations
    create_visualizations(all_stats, output_dir)
    
    # Suggest fixes
    suggest_fixes(overall_min, overall_max, overall_mean, overall_std)
    
    print(f"\n✅ Analysis complete! Check {output_dir} for visualizations")

if __name__ == "__main__":
    main()
