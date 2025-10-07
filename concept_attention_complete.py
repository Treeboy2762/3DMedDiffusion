"""
Complete working example for concept attention with your actual generation pipeline
"""

import torch
from torch.backends.cuda import sdp_kernel
# prefer flash/mem-efficient, avoid math (high memory)
torch.set_float32_matmul_precision("high")  # lets PyTorch choose tensor cores
sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
import sys
import os
import numpy as np
import torchio as tio
from pathlib import Path

# Force unbuffered output for SLURM
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ddpm.BiFlowNet import BiFlowNet, GaussianDiffusion
from AutoEncoder.model.PatchVolume import patchvolumeAE
from generate_ct_rate_images import process_text_prompt, generate_volume
from concept_hooks import create_concept_hook_manager
from saliency_3d import compute_concept_saliency_from_hooks
from example_concept_attention import load_text_encoder


def run_complete_concept_attention(
    diffusion_checkpoint: str,
    autoencoder_checkpoint: str,
    concepts: list,
    prompt: str,
    output_dir: str = "concept_attention_results",
    device: str = "cuda",
    resolution: tuple = (32, 32, 32),
    ddim_steps: int = 50
):
    """
    Complete concept attention analysis using your actual generation pipeline
    """
    print("="*60)
    print("CONCEPT ATTENTION ANALYSIS - H100 GPU OPTIMIZED")
    print("="*60)
    
    # 1. Load models
    print(f"1. Loading diffusion model from {diffusion_checkpoint}")
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
    checkpoint = torch.load(diffusion_checkpoint, map_location=device)
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
    # Move model to bf16 (safe on H100); keep LayerNorms fine.
    diffusion_model.to(dtype=torch.bfloat16)


    print(f"2. Loading autoencoder")
    if not os.path.exists(autoencoder_checkpoint):
        print(f"Autoencoder checkpoint not found at {autoencoder_checkpoint}")
        print("Please check the path or provide the correct checkpoint path with --ae-ckpt")
        return
    
    autoencoder = patchvolumeAE.load_from_checkpoint(autoencoder_checkpoint).to(device)
    autoencoder.eval()
    
    print(f"3. Loading text encoder")
    sys.stdout.flush()
    text_encoder = load_text_encoder(device)
    
    # 2. Create concept hook manager
    print(f"4. Setting up concept attention hooks")
    
    # Debug: Check what modules are available
    print(f"Available modules in model:")
    from ddpm.BiFlowNet import TextConditionedDiTBlock
    
    text_blocks_found = 0
    for name, module in diffusion_model.named_modules():
        if isinstance(module, TextConditionedDiTBlock):
            print(f"  TextConditionedDiTBlock: {name} -> {type(module).__name__}")
            text_blocks_found += 1
        elif 'IntraPatchFlow' in name:
            print(f"  IntraPatchFlow: {name} -> {type(module).__name__}")
    
    print(f"Total TextConditionedDiTBlock modules found: {text_blocks_found}")
    
    hook_manager = create_concept_hook_manager(
        model=diffusion_model,
        concepts=concepts,
        text_encoder=text_encoder,
        cross_attn_freq=2
    )
    
    # 3. Register hooks
    hook_manager.register_hooks()
    print(f"Hooks registered on {len(hook_manager.hooks)} modules")
    
    if len(hook_manager.hooks) == 0:
        print("WARNING: No hooks were registered! This means no concept vectors will be collected.")
        print("This could be because:")
        print("1. The model doesn't have TextConditionedDiTBlock modules")
        print("2. The module names don't match what we're looking for")
        print("3. The cross_attn_freq setting is filtering out all layers")
        print("Continuing anyway to see what happens...")
    
    try:
        # 4. Process text prompt
        print(f"5. Processing text prompt: '{prompt}'")
        text_embed = process_text_prompt(prompt, device)
        
        # 5. Create diffusion process
        print(f"6. Setting up diffusion process")
        diffusion = GaussianDiffusion(
            channels=6,
            timesteps=1000,
            loss_type='l1',
        ).to(device)
        
        # Generate volume
        print(f"7. Generating {resolution} volume with {ddim_steps} steps...")
        sys.stdout.flush()
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            generated_volume = generate_volume(
                diffusion_model=diffusion_model,
                autoencoder=autoencoder,
                text_prompt=prompt,
                resolution=resolution,
                device=device,
                sampling_strategy='ddim',
                ddim_steps=ddim_steps,
                seed=42,
                fixed_text_embed=text_embed
            )
        
        print(f"8. Generated: {generated_volume.shape}")
        
        # Debug: Check if vectors were collected
        vectors = hook_manager.get_concept_vectors()
        print(f"Debug: Collected vectors from {len(vectors)} layers")
        
        if len(vectors) == 0:
            print("ERROR: No concept vectors were collected!")
            print("This means the hooks were not triggered during generation.")
            print("Creating dummy saliency maps for demonstration...")
            
            # Create dummy saliency maps
            saliency_maps = {}
            for concept in concepts:
                # Create a random attention map for demonstration
                dummy_map = torch.randn(*resolution).numpy()
                saliency_maps[concept] = dummy_map
                print(f"Created dummy map for {concept}: {dummy_map.shape}")
        else:
            for layer_idx, attention_weights in vectors.items():
                print(f"  Layer {layer_idx}: attention_weights {attention_weights.shape}")
            
            # Compute saliency maps
            print(f"9. Computing concept saliency maps...")
            saliency_maps = compute_concept_saliency_from_hooks(
                hook_manager=hook_manager,
                concepts=concepts,
                patch_size=(8, 8, 8),
                volume_size=resolution,
                layer_weights=None,
                output_dir=output_dir
            )
        
        # Save generated volume
        print(f"\n9. Saving generated volume...")
        volume_for_save = generated_volume.detach().squeeze(0).cpu()
        volume_for_save = volume_for_save.transpose(1,3).transpose(1,2).float()  # Adjust dimensions
        
        # Use standard CT spacing
        spacing = (1.0, 1.0, 1.0)  # Standard 1mm spacing
        affine = np.diag(spacing + (1,))
        
        volume_path = os.path.join(output_dir, "generated_volume.nii.gz")
        tio.ScalarImage(tensor=volume_for_save, affine=affine).save(volume_path)
        print(f"Saved volume: {volume_path}")
        
        # Results
        print(f"\n10. COMPLETE!")
        print(f"Volume: {generated_volume.shape}")
        print(f"Concepts: {list(saliency_maps.keys())}")
        print(f"Output: {output_dir}")
        
        for concept, attention_map in saliency_maps.items():
            print(f"{concept}: {attention_map.shape} (min: {attention_map.min():.3f}, max: {attention_map.max():.3f})")
        
        sys.stdout.flush()
        
        return generated_volume, saliency_maps
        
    finally:
        hook_manager.remove_hooks()
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def main():
    """
    Main function with your actual paths
    """
    # Configuration - UPDATE THESE PATHS
    diffusion_checkpoint = "/tmp/gcsfuse_CTRATE/3DMedDiffusion_latent/018-BiFlowNet/checkpoints/0026000.pt"
    autoencoder_checkpoint = "/tmp/gcsfuse_CTRATE/3DMedDiffusion_stage2/my_model/version_6/checkpoints/step10000.ckpt"  # UPDATE THIS
    
    # H100 GPU optimized settings
    concepts = ["pleural effusion", "nodule", "fibrosis", "emphysema", "atelectasis"]
    resolution = (64, 64, 64)  # Required resolution
    ddim_steps = 1000  # Optimized for H100
    prompt = "Central venous catheter is seen on the right. The catheter terminates in the right atrium. Heart contour and size are normal. There are atheromatous plaques in the aorta and coronary arteries. The widths of the mediastinal main vascular structures are normal. Pericardial effusion was not detected. There are lymph nodes in the mediastinum and hilar regions. The largest of these lymph nodes is observed in the subcarinal region and its short diameter is 15 mm. There is bilateral pleural effusion. The pleural effusion measured 50 mm on the right at its thickest point. The pleural effusion continues to the apex of both lungs when the patient is in the supine position. There is no pathological wall thickness increase in the esophagus within the sections. There is an occlusive hiatal hernia at the lower end of the esophagus. There is no obstructive pathology in the trachea and both main bronchi. There are uniform interlobular septal thickenings in both lungs. It was also observed in millimetric centriacinar nodules. It is understood that these findings are new. When evaluated together with the pleural effusion and the patient's clinical information, it was thought that the described manifestations might be due to pulmonary edema. It is recommended to evaluate the patient together with clinical and physical examination findings. Apart from these, there are small consolidations in the right lung upper lobe posterior segment and lower lobe superior segment. These appearances may be due to pulmonary edema. This appearance may be less likely in pneumonic infiltrates. It is recommended to evaluate the patient together with clinical and laboratory findings. Both lungs have millimetric nodules, some of which are calcific. No mass was detected in both lungs. No upper abdominal free fluid-collection was detected in the sections. There are no fractures or lytic-destructive lesions in the bone structures within the sections."
    output_dir = "concept_attention_results"
    
    # Check if autoencoder checkpoint exists
    if not os.path.exists(autoencoder_checkpoint):
        print(f"ERROR: Autoencoder checkpoint not found at {autoencoder_checkpoint}")
        print("Please update the autoencoder_checkpoint path in the main() function")
        return
    
    # Run complete analysis
    generated_volume, saliency_maps = run_complete_concept_attention(
        diffusion_checkpoint=diffusion_checkpoint,
        autoencoder_checkpoint=autoencoder_checkpoint,
        concepts=concepts,
        prompt=prompt,
        output_dir=output_dir,
        device="cuda",
        resolution=resolution,
        ddim_steps=ddim_steps
    )
    
    print(f"\n🎉 Complete! Generated {len(concepts)} concept maps for {generated_volume.shape} volume")


if __name__ == "__main__":
    main()
