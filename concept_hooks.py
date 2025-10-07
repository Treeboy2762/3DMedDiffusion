"""
Forward hooks for extracting concept attention from TextConditionedDiTBlock
without modifying the trained model architecture.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from einops import rearrange


class ConceptAttentionHook:
    """
    Forward hook that computes concept attention using existing cross-attention weights
    """
    
    def __init__(self, concepts: List[str], text_encoder, layer_idx: int):
        self.concepts = concepts
        self.text_encoder = text_encoder
        self.layer_idx = layer_idx
        # Instead of storing all vectors, accumulate attention weights directly
        self.accumulated_attention = None
        self.accumulated_weights = 0.0
        self.num_samples = 0
        
    def __call__(self, module, input, output):
        """Hook function called during CrossAttention forward pass"""
        print(f"DEBUG: CrossAttention hook called on layer {self.layer_idx}")
        print(f"DEBUG: Input length: {len(input)}")
        
        # Debug: Let's see what we actually get
        print(f"DEBUG: input type: {type(input)}")
        print(f"DEBUG: input length: {len(input)}")
        for i, item in enumerate(input):
            print(f"DEBUG: input[{i}] type: {type(item)}, shape: {getattr(item, 'shape', 'N/A')}")
        
        # CrossAttention forward signature: (x, context)
        # x = modulated image features, context = text_context (864 dims = time + text + res)
        if len(input) >= 2:
            x, text_context = input[0], input[1]
            print(f"DEBUG: CrossAttention inputs: x={x.shape}, text_context={text_context.shape}")
            
            # Extract text conditioning from the 864-dimensional tensor
            # text_context shape: [512, 864] = [batch, time_cond(288) + text_cond(288) + res_cond(288)]
            text_cond = text_context[:, 288:576]  # Extract text conditioning (288 dims)
            print(f"DEBUG: Extracted text_cond shape: {text_cond.shape}")
            
            # Project text conditioning to match image feature dimension (72)
            text_cond_projected = self._project_text_conditioning(text_cond, x.shape[-1])
            print(f"DEBUG: Projected text_cond shape: {text_cond_projected.shape}")
            
            # Add batch dimension: [512, 72] -> [1, 512, 72]
            text_cond_projected = text_cond_projected.unsqueeze(0)
            print(f"DEBUG: After adding batch dim: {text_cond_projected.shape}")
        else:
            print(f"DEBUG: Not enough arguments: {len(input)}")
            return
        
        print(f"DEBUG: x shape: {x.shape}")
        print(f"DEBUG: text_context shape: {text_context.shape}")
        print(f"DEBUG: Got [batch={x.shape[0]}, seq_len={x.shape[1]}, dim={x.shape[2]}]")
        
        # Check what resolution this layer is operating on
        actual_tokens = x.shape[1]
        print(f"DEBUG: Actual tokens: {actual_tokens:,}")
        
        if text_context is None:
            print(f"DEBUG: text_context is None - returning early")
            return
        
        # We're in the TextConditionedDiTBlock, need to access its cross_attn module
        cross_attn = module.cross_attn
        print(f"DEBUG: Accessing cross_attn module: {type(cross_attn).__name__}")
        
        # Encode concepts using the same text encoder
        concept_embeddings = self._encode_concepts()
        
        # Project concepts to match image feature dimensions (72)
        concept_context = self._project_concepts(concept_embeddings, x.shape[-1])
        
        # Compute concept queries, keys, values using existing weights
        concept_q = cross_attn.to_q(concept_context)
        concept_k = cross_attn.to_k(concept_context)
        concept_v = cross_attn.to_v(concept_context)
        
        # Get image queries (for saliency computation)
        img_q = cross_attn.to_q(x)
        print('img_q', img_q.shape)
        
        # Reshape for multi-head attention
        heads = cross_attn.heads
        concept_q = rearrange(concept_q, 'b n (h d) -> b h n d', h=heads)
        concept_k = rearrange(concept_k, 'b n (h d) -> b h n d', h=heads)
        concept_v = rearrange(concept_v, 'b n (h d) -> b h n d', h=heads)
        img_q = rearrange(img_q, 'b n (h d) -> b h n d', h=heads)
        
        # Get text context keys and values using the projected text conditioning
        text_k = cross_attn.to_k(text_cond_projected)
        text_v = cross_attn.to_v(text_cond_projected)

        print('text context projection shapes', text_cond_projected.shape, text_k.shape)
        text_k = rearrange(text_k, 'b n (h d) -> b h n d', h=heads)
        text_v = rearrange(text_v, 'b n (h d) -> b h n d', h=heads)
        
        # Concatenate concept and text keys/values
        BP = text_k.size(0)
        if concept_k.size(0) == 1:
            concept_k = concept_k.expand(BP, -1, -1, -1).contiguous()
            concept_v = concept_v.expand(BP, -1, -1, -1).contiguous()

        combined_k = torch.cat([concept_k, text_k], dim=2)
        combined_v = torch.cat([concept_v, text_v], dim=2)
        
        # Compute concept attention (concepts attend to both concepts and text)
        concept_attention = F.scaled_dot_product_attention(
            concept_q, combined_k, combined_v,
            dropout_p=0.0, is_causal=False
        )
        
        # Reshape back
        concept_attention = rearrange(concept_attention, 'b h n d -> b n (h d)')
        img_q_flat = rearrange(img_q, 'b h n d -> b n (h d)')
        print('img_q_flat', img_q_flat.shape)
        
        # Compute attention weights immediately and accumulate
        # concept_attention: [batch, num_concepts, dim] 
        # img_q_flat: [batch, num_patches, dim]
        
        print(f"DEBUG: concept_attention shape: {concept_attention.shape}")
        print(f"DEBUG: img_q_flat shape: {img_q_flat.shape}")
        
        # Normalize for cosine similarity
        concept_vec = F.normalize(concept_attention, dim=-1)  # [batch, num_concepts, dim]
        img_vec = F.normalize(img_q_flat, dim=-1)           # [batch, num_patches, dim]

        # Compute attention weights: [batch, num_concepts, num_patches]
        attention_weights = torch.einsum(
            'bcd,bpd->bcp', concept_vec, img_vec
        )  # [batch, num_concepts, num_patches]
        
        # Average across batch dimension
        attention_weights = attention_weights.mean(dim=0)  # [num_concepts, num_patches]
        
        # Apply softmax across spatial dimension (patches)
        attention_weights = F.softmax(attention_weights, dim=-1)  # [num_concepts, num_patches]
        
        # Accumulate attention weights (running average)
        if self.accumulated_attention is None:
            self.accumulated_attention = attention_weights.detach().cpu()
            # Store the detected resolution for this layer
        else:
            self.accumulated_attention = (
                (self.accumulated_attention * self.num_samples + attention_weights.detach().cpu()) 
                / (self.num_samples + 1)
            )
        
        self.num_samples += 1
        
        print(f"DEBUG: Accumulated attention weights - shape: {attention_weights.shape}, samples: {self.num_samples}")
    
    def _encode_concepts(self) -> torch.Tensor:
        """Encode disease concepts using medical BERT tokenizer and encoder"""
        # Format concepts as medical descriptions
        concept_texts = self.concepts
        
        # Use the medical text encoder's encode_batch method for efficiency
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
        """Project concept embeddings to target dimension"""
        if hasattr(self.text_encoder, 'projection'):
            return self.text_encoder.projection(concept_embeddings)
        else:
            # Simple linear projection
            device = concept_embeddings.device
            projection = torch.nn.Linear(concept_embeddings.shape[-1], target_dim).to(device)
            return projection(concept_embeddings)
    
    def _project_text_conditioning(self, text_cond: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Project text conditioning from 288 dimensions to target dimension (72)"""
        device = text_cond.device
        projection = torch.nn.Linear(text_cond.shape[-1], target_dim).to(device)
        return projection(text_cond)


class ConceptHookManager:
    """
    Manages concept attention hooks across multiple TextConditionedDiTBlock layers
    """
    
    def __init__(self, model, concepts: List[str], text_encoder, cross_attn_freq: int = 2):
        self.model = model
        self.concepts = concepts
        self.text_encoder = text_encoder
        self.cross_attn_freq = cross_attn_freq
        self.hooks = []
        self.concept_hooks = {}
        
        # Extract patch size from model if available
        self.patch_size = getattr(model, 'patch_size', None)
        if self.patch_size is None:
            # Try to find it in the model's __init__ parameters or attributes
            for attr_name in ['patch_size', 'patch_size_3d']:
                if hasattr(model, attr_name):
                    self.patch_size = getattr(model, attr_name)
                    break
        
        print(f"DEBUG: Model patch size: {self.patch_size}")
        
    def register_hooks(self):
        """Register concept attention hooks on TextConditionedDiTBlock modules"""
        from ddpm.BiFlowNet import TextConditionedDiTBlock
        
        hook_count = 0
        for name, module in self.model.named_modules():
            # Hook directly into TextConditionedDiTBlock modules
            if isinstance(module, TextConditionedDiTBlock):
                layer_idx = self._extract_layer_index(name)
                print(f"Found TextConditionedDiTBlock: {name} (layer {layer_idx})")
                print(f"  -> Module type: {type(module).__name__}")
                print(f"  -> Full path: {name}")
                
                if layer_idx is not None and layer_idx % self.cross_attn_freq == 0:
                    self._register_hook_on_text_dit_block(module, layer_idx)
                    hook_count += 1
        
        print(f"Total hooks registered: {hook_count}")
        
    def _register_hook_on_text_dit_block(self, module, layer_idx):
        concept_hook = ConceptAttentionHook(self.concepts, self.text_encoder, layer_idx)
        concept_hook.patch_size = self.patch_size  # Pass patch size to hook
        hook = module.register_forward_pre_hook(concept_hook, with_kwargs=True)
        self.hooks.append(hook)
        self.concept_hooks[layer_idx] = concept_hook
        print(f"    -> Pre-hook registered on TextConditionedDiTBlock layer {layer_idx}")


    def _extract_layer_index(self, layer_name: str) -> Optional[int]:
        """Extract layer index from module name"""
        try:
            parts = layer_name.split('.')
            
            # Create unique layer index based on the full path
            if 'IntraPatchFlow_input' in layer_name:
                # For input layers: 0, 1, 2, etc.
                for part in parts:
                    if part.isdigit():
                        return int(part)
            elif 'IntraPatchFlow_mid' in layer_name:
                # For middle layers: 100, 101, 102, etc.
                for part in parts:
                    if part.isdigit():
                        return 100 + int(part)
            elif 'IntraPatchFlow_output' in layer_name:
                # For output layers: 200, 201, 202, etc.
                for part in parts:
                    if part.isdigit():
                        return 200 + int(part)
        except:
            pass
        return None
    
    def get_concept_vectors(self) -> Dict[int, torch.Tensor]:
        """Get accumulated attention weights from all layers"""
        vectors = {}
        for layer_idx, hook in self.concept_hooks.items():
            if hook.accumulated_attention is not None:
                vectors[layer_idx] = hook.accumulated_attention  # [num_concepts, num_patches]
        return vectors
    
    def get_concept_vectors(self):
        """Get accumulated concept attention vectors from all layers"""
        vectors = {}
        for layer_idx, hook in self.concept_hooks.items():
            if hook.accumulated_attention is not None:
                # Return the accumulated attention weights
                vectors[layer_idx] = hook.accumulated_attention
        return vectors
    
    def clear_vectors(self):
        """Clear accumulated attention weights"""
        for hook in self.concept_hooks.values():
            hook.accumulated_attention = None
            hook.accumulated_weights = 0.0
            hook.num_samples = 0
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.concept_hooks = {}


def create_concept_hook_manager(
    model, 
    concepts: List[str], 
    text_encoder,
    cross_attn_freq: int = 2
) -> ConceptHookManager:
    """
    Factory function to create concept hook manager
    """
    return ConceptHookManager(
        model=model,
        concepts=concepts,
        text_encoder=text_encoder,
        cross_attn_freq=cross_attn_freq
    )
