"""
3D Saliency map computation for concept attention in volumetric medical images
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import nibabel as nib
from pathlib import Path


class SaliencyMap3D:
    """
    Computes and visualizes 3D saliency maps from concept attention vectors
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 2, 2),
        volume_size: Tuple[int, int, int] = (8, 8, 8),
        upsample_mode: str = 'trilinear'
    ):
        self.patch_size = patch_size
        self.volume_size = volume_size
        self.upsample_mode = upsample_mode
        
        # Compute patch grid dimensions
        self.patches_per_dim = [
            vol_size // patch_size 
            for vol_size, patch_size in zip(volume_size, patch_size)
        ]
        self.num_patches = np.prod(self.patches_per_dim)
        
    def compute_attention_heatmaps(
        self, 
        concept_vectors: Dict[int, torch.Tensor],
        image_vectors: Dict[int, torch.Tensor],
        concepts: List[str],
        layer_weights: Optional[Dict[int, float]] = None,
        timestep_weights: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute 3D attention heatmaps from concept and image vectors
        
        Args:
            concept_vectors: Dict[layer_idx, tensor] - concept attention vectors
            image_vectors: Dict[layer_idx, tensor] - image output vectors  
            concepts: List of concept names
            layer_weights: Optional weights for different layers
            timestep_weights: Optional weights for different timesteps
            
        Returns:
            Dict[concept_name, 3D_attention_map]
        """
        if layer_weights is None:
            layer_weights = {layer: 1.0 for layer in concept_vectors.keys()}
        
        # Collect attention weights from all layers
        all_attention_weights = []
        all_weights = []
        
        for layer_idx in concept_vectors.keys():
            concept_vecs = concept_vectors[layer_idx]  # [timesteps, B, num_concepts, dim]
            image_vecs = image_vectors[layer_idx]      # [timesteps, B, num_patches, dim]
            
            # Average across batch dimension
            if concept_vecs.dim() > 3:
                concept_vecs = concept_vecs.mean(dim=1)  # [timesteps, num_concepts, dim]
                image_vecs = image_vecs.mean(dim=1)      # [timesteps, num_patches, dim]
            
            # Compute attention weights for each timestep
            for t in range(concept_vecs.shape[0]):
                # Dot product between concept and image vectors
                attention_weights = torch.einsum(
                    'cd,pd->cp', concept_vecs[t], image_vecs[t]
                )  # [num_concepts, num_patches]
                
                # Apply softmax across spatial dimension
                attention_weights = F.softmax(attention_weights, dim=-1)
                
                # Weight by layer and timestep
                weight = layer_weights[layer_idx]
                if timestep_weights is not None and t < len(timestep_weights):
                    weight *= timestep_weights[t]
                
                all_attention_weights.append(attention_weights)
                all_weights.append(weight)
        
        # Average across all layers and timesteps
        if all_attention_weights:
            stacked_weights = torch.stack(all_attention_weights)  # [total_samples, num_concepts, num_patches]
            weights_tensor = torch.tensor(all_weights).view(-1, 1, 1)  # [total_samples, 1, 1]
            
            # Weighted average
            weighted_attention = (stacked_weights * weights_tensor).sum(dim=0) / weights_tensor.sum()
        else:
            raise ValueError("No attention weights collected")
        
        # Convert to 3D saliency maps
        saliency_maps = {}
        for i, concept in enumerate(concepts):
            patch_attention = weighted_attention[i]  # [num_patches]
            volume_attention = self._patches_to_volume(patch_attention)
            saliency_maps[concept] = volume_attention
            
        return saliency_maps
    
    def compute_saliency_from_attention_weights(
        self, 
        attention_weights: Dict[int, torch.Tensor],
        concepts: List[str],
        layer_weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute 3D saliency maps directly from accumulated attention weights
        
        Args:
            attention_weights: Dict[layer_idx, tensor] - accumulated attention weights [num_concepts, num_patches]
            concepts: List of concept names
            layer_weights: Optional weights for different layers
            
        Returns:
            Dict[concept_name, 3D_attention_map]
        """
        if layer_weights is None:
            layer_weights = {layer: 1.0 for layer in attention_weights.keys()}
        
        # Average attention weights across layers
        total_weight = 0.0
        weighted_attention = None
        
        for layer_idx, attention in attention_weights.items():
            weight = layer_weights[layer_idx]
            
            if weighted_attention is None:
                weighted_attention = attention * weight
            else:
                weighted_attention += attention * weight
            
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_attention = weighted_attention / total_weight
        else:
            raise ValueError("No valid attention weights found")
        
        # Convert to 3D saliency maps
        saliency_maps = {}
        for i, concept in enumerate(concepts):
            patch_attention = weighted_attention[i]  # [num_patches]
            volume_attention = self._patches_to_volume(patch_attention)
            saliency_maps[concept] = volume_attention
            
        return saliency_maps
    
    def _patches_to_volume(self, patch_attention: torch.Tensor) -> np.ndarray:
        """
        Convert patch-level attention to 3D volume
        """
        # Reshape from flattened patches to 3D patch grid
        volume_attention = patch_attention.reshape(self.patches_per_dim)  # [D, H, W]
        
        # Upsample to full resolution
        volume_attention = F.interpolate(
            volume_attention.unsqueeze(0).unsqueeze(0),  # [1, 1, D, H, W]
            size=self.volume_size,
            mode=self.upsample_mode,
            align_corners=False
        ).squeeze(0).squeeze(0)  # [D, H, W]
        
        return volume_attention.detach().cpu().numpy()
    
    def normalize_saliency_maps(
        self, 
        saliency_maps: Dict[str, np.ndarray],
        normalization: str = 'minmax'
    ) -> Dict[str, np.ndarray]:
        """
        Normalize saliency maps for visualization
        
        Args:
            saliency_maps: Dict of concept attention maps
            normalization: 'minmax', 'zscore', or 'none'
        """
        normalized_maps = {}
        
        if normalization == 'minmax':
            # Normalize each map to [0, 1]
            for concept, attention_map in saliency_maps.items():
                min_val = attention_map.min()
                max_val = attention_map.max()
                if max_val > min_val:
                    normalized_map = (attention_map - min_val) / (max_val - min_val)
                else:
                    normalized_map = attention_map
                normalized_maps[concept] = normalized_map
                
        elif normalization == 'zscore':
            # Z-score normalization
            for concept, attention_map in saliency_maps.items():
                mean_val = attention_map.mean()
                std_val = attention_map.std()
                if std_val > 0:
                    normalized_map = (attention_map - mean_val) / std_val
                else:
                    normalized_map = attention_map
                normalized_maps[concept] = normalized_map
                
        else:  # 'none'
            normalized_maps = saliency_maps
            
        return normalized_maps
    
    def save_saliency_maps(
        self, 
        saliency_maps: Dict[str, np.ndarray],
        output_dir: str,
        normalize: bool = True,
        format: str = 'nifti'
    ):
        """
        Save saliency maps to files
        
        Args:
            saliency_maps: Dict of concept attention maps
            output_dir: Output directory path
            normalize: Whether to normalize maps before saving
            format: 'nifti' or 'numpy'
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Normalize if requested
        if normalize:
            saliency_maps = self.normalize_saliency_maps(saliency_maps, 'minmax')
        
        for concept, attention_map in saliency_maps.items():
            if format == 'nifti':
                # Save as NIfTI file
                nii_img = nib.Nifti1Image(attention_map, affine=np.eye(4))
                filename = f"concept_attention_{concept}.nii.gz"
                nib.save(nii_img, output_path / filename)
                
            elif format == 'numpy':
                # Save as numpy array
                filename = f"concept_attention_{concept}.npy"
                np.save(output_path / filename, attention_map)
                
            print(f"Saved concept attention map: {output_path / filename}")


def create_saliency_computer(
    patch_size: Tuple[int, int, int] = (2, 2, 2),
    volume_size: Tuple[int, int, int] = (8, 8, 8)
) -> SaliencyMap3D:
    """
    Factory function to create saliency map computer
    """
    return SaliencyMap3D(patch_size=patch_size, volume_size=volume_size)


def compute_concept_saliency_from_hooks(
    hook_manager,
    concepts: List[str],
    patch_size: Tuple[int, int, int] = (2, 2, 2),
    volume_size: Tuple[int, int, int] = (8, 8, 8),
    layer_weights: Optional[Dict[int, float]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Convenience function to compute saliency maps from hook manager
    
    Args:
        hook_manager: ConceptHookManager instance
        concepts: List of concept names
        patch_size: 3D patch size
        volume_size: Full 3D volume size
        layer_weights: Optional layer weights
        output_dir: Optional output directory for saving
        
    Returns:
        Dict of concept saliency maps
    """
    # Get accumulated attention weights from hooks
    attention_weights = hook_manager.get_concept_vectors()
    
    if not attention_weights:
        raise ValueError("No attention weights collected from hooks")
    
    # Create saliency computer
    saliency_computer = create_saliency_computer(patch_size, volume_size)
    
    # Compute saliency maps directly from accumulated attention weights
    saliency_maps = saliency_computer.compute_saliency_from_attention_weights(
        attention_weights=attention_weights,
        concepts=concepts,
        layer_weights=layer_weights
    )
    
    # Save if output directory provided
    if output_dir:
        saliency_computer.save_saliency_maps(saliency_maps, output_dir)
    
    return saliency_maps


