"""
Example script for running concept attention analysis on 3D medical diffusion model
"""

import torch
import sys
import os
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ddpm.BiFlowNet import BiFlowNet
from concept_hooks import create_concept_hook_manager
from saliency_3d import compute_concept_saliency_from_hooks


def load_trained_model(checkpoint_path: str, device: str = "cuda"):
    """
    Load your trained 3DMedDiffusion model
    """
    # Initialize model with your training parameters
    model = BiFlowNet(
        dim=768,
        text_embed_dim=768,
        dim_mults=(1, 1, 2, 4, 8),
        sub_volume_size=(8, 8, 8),
        patch_size=2,
        channels=3,
        attn_heads=8,
        DiT_num_heads=8,
        cross_attn_freq=2
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    return model


def load_text_encoder(device: str = "cuda"):
    """
    Load your Microsoft BiomedVLP-CXR-BERT-specialized text encoder and tokenizer
    """
    from transformers import AutoModel, AutoTokenizer
    
    # Load the same text encoder and tokenizer as in your training
    text_encoder = AutoModel.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", 
        trust_remote_code=True, 
        torch_dtype=torch.float32
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", 
        trust_remote_code=True
    )
    
    # Freeze text encoder
    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()
    
    # Create wrapper class for easy use
    class MedicalTextEncoder:
        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            
        def encode(self, text):
            """Encode text using the medical BERT model"""
            with torch.no_grad():
                # Tokenize the text
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Use [CLS] token embedding (first token)
                # This matches how you likely encode text in your training
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
                
                return cls_embedding.squeeze(0)  # [hidden_dim]
        
        def encode_batch(self, texts):
            """Encode multiple texts at once"""
            with torch.no_grad():
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
                
                return cls_embeddings
    
    return MedicalTextEncoder(text_encoder, tokenizer, device)


def run_concept_attention_analysis(
    checkpoint_path: str,
    concepts: list,
    prompt: str,
    output_dir: str = "concept_attention_results",
    device: str = "cuda"
):
    """
    Run concept attention analysis on a trained model
    """
    print(f"Loading model from {checkpoint_path}")
    model = load_trained_model(checkpoint_path, device)
    
    print(f"Loading text encoder")
    text_encoder = load_text_encoder(device)
    
    print(f"Creating concept hook manager")
    hook_manager = create_concept_hook_manager(
        model=model,
        concepts=concepts,
        text_encoder=text_encoder,
        cross_attn_freq=2
    )
    
    print(f"Registering hooks")
    hook_manager.register_hooks()
    
    try:
        print(f"Running generation with concept attention")
        
        # Import your generation function
        from generate_ct_rate_images import generate_volume, process_text_prompt
        from ddpm.BiFlowNet import GaussianDiffusion
        from AutoEncoder.model.PatchVolume import patchvolumeAE
        
        # Load autoencoder (needed for generation)
        print("Loading autoencoder...")
        autoencoder = patchvolumeAE(
            input_nc=1,
            output_nc=1, 
            nf=64,
            codebook_size=1024,
            codebook_dim=256,
            downsampling_factor=4
        )
        
        # Load autoencoder checkpoint (you'll need to provide the path)
        # autoencoder_ckpt = torch.load("path/to/autoencoder/checkpoint.pth")
        # autoencoder.load_state_dict(autoencoder_ckpt)
        autoencoder = autoencoder.to(device)
        autoencoder.eval()
        
        # Process the text prompt using your existing function
        text_embed = process_text_prompt(prompt, device)
        
        # Create diffusion process
        diffusion = GaussianDiffusion(
            channels=6,  # volume_channels from your config
            timesteps=1000,
            loss_type='l1',
        ).to(device)
        
        # Generate the volume - this will trigger the concept attention hooks
        print(f"Generating volume with prompt: '{prompt}'")
        generated_volume = generate_volume(
            diffusion_model=model,
            autoencoder=autoencoder,
            text_prompt=prompt,
            resolution=(32, 32, 32),
            device=device,
            sampling_strategy='ddim',
            ddim_steps=50,  # Fewer steps for faster generation
            seed=42,
            fixed_text_embed=text_embed  # Use the processed text embedding
        )
        
        print(f"Generated volume shape: {generated_volume.shape}")
        
        print(f"Computing saliency maps")
        saliency_maps = compute_concept_saliency_from_hooks(
            hook_manager=hook_manager,
            concepts=concepts,
            patch_size=(2, 2, 2),
            volume_size=(8, 8, 8),
            layer_weights=None,  # Equal weights for all layers
            output_dir=output_dir
        )
        
        print(f"Analysis complete!")
        print(f"Generated saliency maps for concepts: {list(saliency_maps.keys())}")
        
        return saliency_maps
        
    finally:
        print(f"Cleaning up hooks")
        hook_manager.remove_hooks()


def main():
    """
    Main function demonstrating concept attention analysis
    """
    # Configuration
    checkpoint_path = "/tmp/gcsfuse_CTRATE/3DMedDiffusion_latent/018-BiFlowNet/checkpoints/0026000.pt"  # Update this path
    concepts = ["pneumonia", "nodule", "fibrosis", "emphysema", "atelectasis"]
    prompt = "CT chest showing pneumonia and nodules"
    output_dir = "concept_attention_results"
    
    # Run analysis
    saliency_maps = run_concept_attention_analysis(
        checkpoint_path=checkpoint_path,
        concepts=concepts,
        prompt=prompt,
        output_dir=output_dir
    )
    
    # Print results summary
    print("\n" + "="*50)
    print("CONCEPT ATTENTION ANALYSIS RESULTS")
    print("="*50)
    
    for concept, attention_map in saliency_maps.items():
        print(f"\nConcept: {concept}")
        print(f"  Attention map shape: {attention_map.shape}")
        print(f"  Min attention: {attention_map.min():.4f}")
        print(f"  Max attention: {attention_map.max():.4f}")
        print(f"  Mean attention: {attention_map.mean():.4f}")
        print(f"  Std attention: {attention_map.std():.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
