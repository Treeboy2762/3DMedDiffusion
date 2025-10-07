"""
ConceptAttention for 3D Medical Diffusion Models
Implements concept-based saliency maps using existing cross-attention weights
without modifying the trained model architecture.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from einops import rearrange, repeat
import nibabel as nib
from pathlib import Path


class ConceptAttention3D:
    """
    Concept attention analyzer for 3D medical diffusion models.
    Uses existing cross-attention weights without modifying the trained model.
    """
    
    def __init__(
        self, 
        model, 
        text_encoder,
        concepts: List[str],
        patch_size: Tuple[int, int, int] = (2, 2, 2),
        volume_size: Tuple[int, int, int] = (8, 8, 8),
        cross_attn_freq: int = 2
    ):
        self.model = model
        self.text_encoder = text_encoder
        self.concepts = concepts
        self.patch_size = patch_size
        self.volume_size = volume_size
        self.cross_attn_freq = cross_attn_freq
        
        # Storage for concept attention vectors
        self.concept_vectors = []
        self.image_vectors = []
        self.layer_indices = []
        self.hooks = []
        
        # Compute patch grid dimensions
        self.patches_per_dim = [
            vol_size // patch_size 
            for vol_size, patch_size in zip(volume_size, patch_size)
        ]
        self.num_patches = np.prod(self.patches_per_dim)
        
    def register_hooks(self):
        """Register forward hooks on TextConditionedDiTBlock layers"""
        for name, module in self.model.named_modules():
            if 'TextConditionedDiTBlock' in name:
                hook = module.register_forward_hook(
                    self._create_concept_hook(name, module)
                )
                self.hooks.append(hook)
    
    def _create_concept_hook(self, layer_name: str, block):
        """Create a forward hook that computes concept attention"""
        def hook(module, input, output):
            # Only process layers with cross-attention enabled
            layer_idx = self._extract_layer_index(layer_name)
            if layer_idx is None or layer_idx % self.cross_attn_freq != 0:
                return
                
            # Extract inputs from the block
            x, c, text_context = input[0], input[1], input[2] if len(input) > 2 else None
            
            if text_context is None:
                return
                
            # Compute concept embeddings using existing text encoder
            concept_embeddings = self._encode_concepts()
            concept_context = self._project_concepts(concept_embeddings, text_context.shape[-1])
            
            # Get cross-attention weights from the block
            cross_attn = block.cross_attn
            
            # Compute concept queries, keys, values using existing weights
            concept_q = cross_attn.to_q(concept_context)  # [B, num_concepts, dim]
            concept_k = cross_attn.to_k(concept_context)  # [B, num_concepts, dim] 
            concept_v = cross_attn.to_v(concept_context)  # [B, num_concepts, dim]
            
            # Get image queries, keys, values
            img_q = cross_attn.to_q(x)  # [B, num_patches, dim]
            img_k = cross_attn.to_k(text_context)  # [B, text_len, dim]
            img_v = cross_attn.to_v(text_context)  # [B, text_len, dim]
            
            # Reshape for multi-head attention
            heads = cross_attn.heads
            dim_head = cross_attn.head_dim
            
            concept_q = rearrange(concept_q, 'b n (h d) -> b h n d', h=heads)
            concept_k = rearrange(concept_k, 'b n (h d) -> b h n d', h=heads)
            concept_v = rearrange(concept_v, 'b n (h d) -> b h n d', h=heads)
            
            img_q = rearrange(img_q, 'b n (h d) -> b h n d', h=heads)
            img_k = rearrange(img_k, 'b n (h d) -> b h n d', h=heads)
            img_v = rearrange(img_v, 'b n (h d) -> b h n d', h=heads)
            
            # Concatenate concept and image keys/values for attention computation
            combined_k = torch.cat([concept_k, img_k], dim=2)  # [B, H, num_concepts + text_len, dim_head]
            combined_v = torch.cat([concept_v, img_v], dim=2)  # [B, H, num_concepts + text_len, dim_head]
            
            # One-way attention: concepts attend to both concepts and image/text
            concept_attention = F.scaled_dot_product_attention(
                concept_q, combined_k, combined_v,
                dropout_p=0.0, is_causal=False
            )
            
            # Reshape back to original format
            concept_attention = rearrange(concept_attention, 'b h n d -> b n (h d)')
            
            # Store concept vectors for saliency computation
            self.concept_vectors.append(concept_attention.detach())
            self.image_vectors.append(output.detach())  # Image output from the block
            self.layer_indices.append(layer_idx)
            
        return hook
    
    def _extract_layer_index(self, layer_name: str) -> Optional[int]:
        """Extract layer index from module name"""
        try:
            # Extract number from layer name like "dit_blocks.5.TextConditionedDiTBlock"
            parts = layer_name.split('.')
            for part in parts:
                if part.isdigit():
                    return int(part)
        except:
            pass
        return None
    
    def _encode_concepts(self) -> torch.Tensor:
        """Encode disease concepts using Microsoft BiomedVLP-CXR-BERT-specialized"""
        # Format concepts as medical descriptions
        concept_texts = [f"medical condition: {concept}" for concept in self.concepts]
        
        # Use batch encoding for efficiency
        if hasattr(self.text_encoder, 'encode_batch'):
            concept_embeddings = self.text_encoder.encode_batch(concept_texts)
        else:
            # Fallback to individual encoding
            concept_embeddings = []
            for concept_text in concept_texts:
                embedding = self.text_encoder.encode(concept_text)
                concept_embeddings.append(embedding)
            concept_embeddings = torch.stack(concept_embeddings)
            
        return concept_embeddings.unsqueeze(0)  # [1, num_concepts, embed_dim]
    
    def _project_concepts(self, concept_embeddings: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Project concept embeddings to match text context dimensions"""
        # Use the same projection as text_context in your model
        if hasattr(self.model, 'text_context_proj'):
            return self.model.text_context_proj(concept_embeddings)
        else:
            # Fallback: linear projection to target dimension
            projection = torch.nn.Linear(concept_embeddings.shape[-1], target_dim)
            return projection(concept_embeddings.to(next(self.model.parameters()).device))
    
    def compute_saliency_maps(self, timesteps: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Compute 3D saliency maps for each concept
        """
        if not self.concept_vectors:
            raise ValueError("No concept vectors collected. Run inference with hooks registered first.")
        
        # Stack vectors from all layers and timesteps
        concept_vecs = torch.stack(self.concept_vectors)  # [layers, B, num_concepts, dim]
        image_vecs = torch.stack(self.image_vectors)      # [layers, B, num_patches, dim]
        
        # Average across batch dimension if present
        if concept_vecs.dim() > 3:
            concept_vecs = concept_vecs.mean(dim=1)  # [layers, num_concepts, dim]
            image_vecs = image_vecs.mean(dim=1)      # [layers, num_patches, dim]
        
        # Compute attention heatmaps
        # concept_vecs: [layers, num_concepts, dim]
        # image_vecs: [layers, num_patches, dim]
        attention_weights = torch.einsum(
            'lcd,lpd->lcp', concept_vecs, image_vecs
        )  # [layers, num_concepts, num_patches]
        
        # Apply softmax across spatial dimension
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Average across layers
        attention_weights = attention_weights.mean(dim=0)  # [num_concepts, num_patches]
        
        # Convert to 3D saliency maps
        saliency_maps = {}
        for i, concept in enumerate(self.concepts):
            # Reshape from patches to 3D volume
            patch_attention = attention_weights[i]  # [num_patches]
            volume_attention = patch_attention.reshape(self.patches_per_dim)  # [D, H, W]
            
            # Upsample to full resolution
            volume_attention = F.interpolate(
                volume_attention.unsqueeze(0).unsqueeze(0),  # [1, 1, D, H, W]
                size=self.volume_size,
                mode='trilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)  # [D, H, W]
            
            saliency_maps[concept] = volume_attention.detach().cpu().numpy()
        
        return saliency_maps
    
    def analyze_generation(
        self, 
        prompt: str, 
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        """
        Generate image and compute concept saliency maps
        """
        # Clear previous results
        self.concept_vectors = []
        self.image_vectors = []
        self.layer_indices = []
        
        # Register hooks
        self.register_hooks()
        
        try:
            # Run generation with your existing pipeline
            # This should use your existing generation code
            generated_volume = self._run_generation(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            # Compute saliency maps
            saliency_maps = self.compute_saliency_maps()
            
            return generated_volume, saliency_maps
            
        finally:
            # Clean up hooks
            self.remove_hooks()
    
    def _run_generation(self, prompt: str, num_inference_steps: int, guidance_scale: float) -> torch.Tensor:
        """
        Run generation using your existing pipeline
        """
        from generate_ct_rate_images import generate_volume, process_text_prompt
        from ddpm.BiFlowNet import GaussianDiffusion
        from AutoEncoder.model.PatchVolume import patchvolumeAE
        
        # Load autoencoder
        autoencoder = patchvolumeAE(
            input_nc=1,
            output_nc=1, 
            nf=64,
            codebook_size=1024,
            codebook_dim=256,
            downsampling_factor=4
        )
        # TODO: Load autoencoder checkpoint
        # autoencoder_ckpt = torch.load("path/to/autoencoder/checkpoint.pth")
        # autoencoder.load_state_dict(autoencoder_ckpt)
        autoencoder = autoencoder.to(self.device)
        autoencoder.eval()
        
        # Process text prompt
        text_embed = process_text_prompt(prompt, self.device)
        
        # Create diffusion process
        diffusion = GaussianDiffusion(
            channels=6,
            timesteps=1000,
            loss_type='l1',
        ).to(self.device)
        
        # Generate volume
        generated_volume = generate_volume(
            diffusion_model=self.model,
            autoencoder=autoencoder,
            text_prompt=prompt,
            resolution=self.volume_size,
            device=self.device,
            sampling_strategy='ddim',
            ddim_steps=num_inference_steps,
            seed=42,
            fixed_text_embed=text_embed
        )
        
        return generated_volume
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def save_saliency_maps(self, saliency_maps: Dict[str, np.ndarray], output_dir: str):
        """Save saliency maps as NIfTI files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for concept, attention_map in saliency_maps.items():
            # Create NIfTI image
            nii_img = nib.Nifti1Image(attention_map, affine=np.eye(4))
            
            # Save with concept name
            filename = f"concept_attention_{concept}.nii.gz"
            nib.save(nii_img, output_path / filename)
            
            print(f"Saved concept attention map: {output_path / filename}")


def create_concept_analyzer(
    checkpoint_path: str,
    concepts: List[str],
    device: str = "cuda"
) -> ConceptAttention3D:
    """
    Factory function to create concept analyzer from checkpoint
    """
    # Load your trained model
    model = BiFlowNet(...)  # Initialize with your parameters
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    # Load your text encoder
    text_encoder = load_text_encoder(...)  # Your text encoder loading code
    
    # Create analyzer
    analyzer = ConceptAttention3D(
        model=model,
        text_encoder=text_encoder,
        concepts=concepts
    )
    
    return analyzer


# Example usage
if __name__ == "__main__":
    # Define disease concepts
    concepts = ["pneumonia", "nodule", "fibrosis", "emphysema", "atelectasis"]
    
    # Create analyzer
    analyzer = create_concept_analyzer(
        checkpoint_path="path/to/your/checkpoint.pth",
        concepts=concepts
    )
    
    # Analyze generation
    generated_volume, saliency_maps = analyzer.analyze_generation(
        prompt="CT chest showing pneumonia",
        num_inference_steps=50
    )
    
    # Save results
    analyzer.save_saliency_maps(saliency_maps, "concept_attention_results/")
    
    print("Concept attention analysis complete!")
    print(f"Generated volume shape: {generated_volume.shape}")
    print(f"Concepts analyzed: {list(saliency_maps.keys())}")
