#!/usr/bin/env python3
"""
Text-to-Image Generation Script for CT-RATE Model
Generates 3D CT volumes from text prompts using your trained BiFlowNet model.
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "."))
sys.path.append(project_root)

import torch
import torch.nn.functional as F
import argparse
import numpy as np
import torchio as tio
from pathlib import Path
from tqdm import tqdm

# Import your model components
from ddpm.BiFlowNet import GaussianDiffusion, BiFlowNet
from AutoEncoder.model.PatchVolume import patchvolumeAE

# Text processing (using BiomedVLP-CXR-BERT like in training)
def process_text_prompt(text_prompt, device, max_text_len=512):
    """
    Process text prompt into embeddings using BiomedVLP-CXR-BERT.
    This matches your training setup exactly - using float32 to match model expectations.
    """
    from transformers import AutoModel, AutoTokenizer
    
    # Load the same text encoder used in training
    text_encoder = AutoModel.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", 
        trust_remote_code=True, 
        torch_dtype=torch.float32  # Use float32 to match model expectations
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", 
        trust_remote_code=True
    )
    
    # Freeze text encoder
    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()
    
    # Process text
    if not text_prompt or text_prompt.strip() == "":
        text_prompt = "Normal chest CT scan."
    
    # Tokenize text
    inputs = tokenizer(
        text_prompt, 
        max_length=max_text_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get text embeddings
    with torch.no_grad():
        outputs = text_encoder(**inputs)
        # Use [CLS] token (first token) from last hidden state
        text_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (1, 768)
        # Convert to float32 to match model expectations
        text_embedding = text_embedding.float()
    
    return text_embedding

def generate_volume(
    diffusion_model, 
    autoencoder, 
    text_prompt="Normal chest CT scan.", 
    resolution=(32, 32, 32),
    device='cuda',
    sampling_strategy='ddpm',
    ddim_steps=100,  # Back to DDIM with 100 steps
    seed=42,
    fixed_z=None,
    fixed_text_embed=None
):
    """
    Generate a 3D volume from text prompt using radiologist report conditioning
    
    Args:
        diffusion_model: Trained BiFlowNet model
        autoencoder: Trained PatchVolume autoencoder
        text_prompt: Radiologist report text description
        resolution: Output resolution tuple
        device: Device to run on
        sampling_strategy: 'ddpm' or 'ddim'
        ddim_steps: Number of DDIM steps (if using DDIM)
        seed: Random seed for reproducibility
    """
    
    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Use fixed text embedding if provided, otherwise process text prompt
    if fixed_text_embed is not None:
        print(f"Using fixed text embedding for prompt: '{text_prompt}'")
        text_embed = fixed_text_embed
    else:
        print(f"Processing text prompt: '{text_prompt}'")
        text_embed = process_text_prompt(text_prompt, device)
    
    # Create diffusion process
    diffusion = GaussianDiffusion(
        channels=6,  # volume_channels from your config
        timesteps=1000,
        loss_type='l1',
    ).to(device)
    
    # Use fixed noise if provided, otherwise generate random noise
    if fixed_z is not None:
        print("Using fixed noise for consistent generation")
        z = fixed_z
    else:
        print("Generating random noise")
        z = torch.randn(1, 6, resolution[0], resolution[1], resolution[2], device=device)
    
    # Resolution embedding (normalized by 64)
    res_emb = torch.tensor(resolution, device=device) / 64.0
    
    print(f"Generating volume with resolution {resolution}")
    print(f"Using {sampling_strategy} sampling strategy...")
    
    # Generate latent codes using diffusion
    with torch.no_grad():
        samples = diffusion.sample(
            diffusion_model, 
            z, 
            y=text_embed,  # Use text embedding instead of class
            res=res_emb, 
            strategy=sampling_strategy,
            ddim_steps=ddim_steps
        )
    
    # Convert from [-1, 1] to codebook range
    samples = (((samples + 1.0) / 2.0) * 
               (autoencoder.codebook.embeddings.max() - autoencoder.codebook.embeddings.min())) + \
               autoencoder.codebook.embeddings.min()
    
    # Decode latent codes to volume
    print("Decoding latent codes to volume...")
    print(f"DEBUG: samples shape: {samples.shape}")
    print(f"DEBUG: autoencoder.codebook.embeddings shape: {autoencoder.codebook.embeddings.shape}")
    print(f"DEBUG: autoencoder.codebook.n_codes: {autoencoder.codebook.n_codes}")
    print(f"DEBUG: autoencoder.codebook.embedding_dim: {autoencoder.codebook.embedding_dim}")
    
    # Verify dimension compatibility
    if samples.shape[1] != autoencoder.codebook.embedding_dim:
        raise ValueError(f"Dimension mismatch: diffusion model generates {samples.shape[1]} channels, "
                        f"but autoencoder expects {autoencoder.codebook.embedding_dim} embedding dimensions")
    
    if resolution[0] * resolution[1] * resolution[2] <= 32*32*32:
        volume = autoencoder.decode(samples, quantize=True)
    else:
        volume = autoencoder.decode_sliding(samples, quantize=True)
    
    return volume

def main():
    parser = argparse.ArgumentParser(description='Generate CT volumes from text prompts')
    
    # Model paths
    parser.add_argument('--ae-ckpt', type=str, 
                       default='/tmp/gcsfuse_CTRATE/3DMedDiffusion_stage2/my_model/version_6/checkpoints/latest_checkpoint.ckpt',
                       help='Path to autoencoder checkpoint')
    parser.add_argument('--diffusion-ckpt', type=str, required=True,
                       help='Path to diffusion model checkpoint')
    
    # Generation parameters
    parser.add_argument('--text-prompt', type=str, 
                       default="Normal chest CT scan with no acute findings.",
                       help='Radiologist report text prompt for generation')
    parser.add_argument('--resolution', type=int, nargs=3, default=[64, 64, 64],
                       help='Output resolution (D H W)')
    parser.add_argument('--output-dir', type=str, default='./generated_volumes',
                       help='Output directory for generated volumes')
    
    # Sampling parameters
    parser.add_argument('--sampling-strategy', type=str, default='ddim', choices=['ddpm', 'ddim'],
                       help='Sampling strategy')
    parser.add_argument('--ddim-steps', type=int, default=100,  # Back to DDIM with 100 steps
                       help='Number of DDIM steps')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load autoencoder
    print("Loading autoencoder...")
    ae_ckpt_path = args.ae_ckpt
    if not os.path.exists(ae_ckpt_path):
        print(f"Autoencoder checkpoint not found at {ae_ckpt_path}")
        print("Please check the path or provide the correct checkpoint path with --ae-ckpt")
        return
    
    autoencoder = patchvolumeAE.load_from_checkpoint(ae_ckpt_path).to(device)
    autoencoder.eval()

    
    print("Autoencoder loaded successfully!")
    
    # Load diffusion model
    print("Loading diffusion model...")
    if not os.path.exists(args.diffusion_ckpt):
        print(f"Diffusion checkpoint not found at {args.diffusion_ckpt}")
        return
    
    # Initialize BiFlowNet model (matching your training config)
    diffusion_model = BiFlowNet(
        dim=72,  # From your training config
        dim_mults=[1,1,2,4,8],
        channels=6,  # volume_channels
        init_kernel_size=3,
        text_embed_dim=768,  # BiomedVLP-CXR-BERT embedding dimension
        learn_sigma=False,
        use_sparse_linear_attn=[0,0,0,1,1],
        vq_size=64,
        num_mid_DiT=1,
        patch_size=1
    ).to(device)
    
    # Load checkpoint - prioritize EMA weights like in training
    checkpoint = torch.load(args.diffusion_ckpt, map_location=device)
    if 'ema' in checkpoint:
        print("Loading EMA weights (recommended for generation)")
        diffusion_model.load_state_dict(checkpoint['ema'], strict=True)
    elif 'model' in checkpoint:
        print("Loading regular model weights (EMA not found)")
        diffusion_model.load_state_dict(checkpoint['model'], strict=True)
    else:
        print("Loading checkpoint directly (no 'ema' or 'model' key found)")
        diffusion_model.load_state_dict(checkpoint, strict=True)
    
    diffusion_model.eval()
    print("Diffusion model loaded successfully!")
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} volume(s)...")
    print(f"Text prompt: '{args.text_prompt}'")
    print(f"Resolution: {args.resolution}")
    
    # Create fixed noise and text embedding for consistent generation (like training)
    print("Setting up fixed noise and text embedding for consistent generation...")
    fixed_z = torch.randn(1, 6, args.resolution[0], args.resolution[1], args.resolution[2], device=device)
    fixed_text_embed = process_text_prompt(args.text_prompt, device)
    
    for i in range(args.num_samples):
        print(f"\nGenerating sample {i+1}/{args.num_samples}...")
        
        # Generate volume using fixed noise and text embedding
        volume = generate_volume(
            diffusion_model=diffusion_model,
            autoencoder=autoencoder,
            text_prompt=args.text_prompt,
            resolution=tuple(args.resolution),
            device=device,
            sampling_strategy=args.sampling_strategy,
            ddim_steps=args.ddim_steps,
            seed=args.seed,  # Use same seed for all samples
            fixed_z=fixed_z,  # Use fixed noise
            fixed_text_embed=fixed_text_embed  # Use fixed text embedding
        )
        
        # Prepare for saving
        volume = volume.detach().squeeze(0).cpu()
        volume = volume.transpose(1,3).transpose(1,2)  # Adjust dimensions
        
        # Use standard CT spacing
        spacing = (1.0, 1.0, 1.0)  # Standard 1mm spacing
        affine = np.diag(spacing + (1,))
        
        # Save volume
        output_name = f"generated_sample_{i+1:03d}.nii.gz"
        output_path = os.path.join(args.output_dir, output_name)
        
        tio.ScalarImage(tensor=volume, affine=affine).save(output_path)
        print(f"Saved: {output_path}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    print(f"\nGeneration complete! Check {args.output_dir} for results.")

if __name__ == "__main__":
    main()
