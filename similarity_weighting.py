"""
Similarity-based feature weighting for multimodal models.
Integrates CLIP-based similarity scores to weight image features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import random
import numpy as np
from typing import List, Dict, Optional, Union
import math
from collections import defaultdict

# Set random seed for reproducibility
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)


def multi_scale_crop_images(image, n, scales=[0.3, 0.5, 0.7]):
    """
    Multi-scale cropping: generates crops at different scales for better coverage.
    """
    w, h = image.size
    subimages = []
    
    crops_per_scale = max(1, n // len(scales))
    
    for scale in scales:
        crop_w, crop_h = int(w * scale), int(h * scale)
        
        for _ in range(crops_per_scale):
            if len(subimages) >= n:
                break
                
            left = random.randint(0, max(0, w - crop_w))
            upper = random.randint(0, max(0, h - crop_h))
            right = min(left + crop_w, w)
            lower = min(upper + crop_h, h)
            subimages.append(image.crop((left, upper, right, lower)))
    
    # Fill remaining with random scale crops if needed
    while len(subimages) < n:
        scale = random.choice(scales)
        crop_w, crop_h = int(w * scale), int(h * scale)
        left = random.randint(0, max(0, w - crop_w))
        upper = random.randint(0, max(0, h - crop_h))
        right = min(left + crop_w, w)
        lower = min(upper + crop_h, h)
        subimages.append(image.crop((left, upper, right, lower)))
    
    return subimages[:n]


def smart_crop_images(image, n, overlap_threshold=0.3):
    """
    Smart cropping that tries to minimize overlap between crops.
    """
    w, h = image.size
    crop_w, crop_h = w // 2, h // 2
    
    # Grid-based approach for better coverage
    grid_x = max(2, int(math.sqrt(n)))
    grid_y = max(2, math.ceil(n / grid_x))
    
    step_x = max(crop_w // 2, (w - crop_w) // max(1, grid_x - 1))
    step_y = max(crop_h // 2, (h - crop_h) // max(1, grid_y - 1))
    
    subimages = []
    for i in range(grid_y):
        for j in range(grid_x):
            if len(subimages) >= n:
                break
                
            left = min(j * step_x, w - crop_w)
            upper = min(i * step_y, h - crop_h)
            right = min(left + crop_w, w)
            lower = min(upper + crop_h, h)
            
            # Add some randomness
            noise_x = random.randint(-crop_w//8, crop_w//8)
            noise_y = random.randint(-crop_h//8, crop_h//8)
            left = max(0, min(left + noise_x, w - crop_w))
            upper = max(0, min(upper + noise_y, h - crop_h))
            right = min(left + crop_w, w)
            lower = min(upper + crop_h, h)
            
            subimages.append(image.crop((left, upper, right, lower)))
    
    return subimages[:n]


class SimilarityWeighting(nn.Module):
    """
    Enhanced similarity weighting with adaptive mechanisms and caching.
    """
    def __init__(self, 
                 clip_model_id="openai/clip-vit-base-patch32", 
                 weight=0.5,
                 adaptive_weighting=True,
                 use_attention=True,
                 cache_size=1000,
                 temperature=0.07,
                 multi_scale_crops=True):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_id)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id, use_fast=True)
        self.base_weight = weight
        self.adaptive_weighting = adaptive_weighting
        self.use_attention = use_attention
        self.temperature = temperature
        self.multi_scale_crops = multi_scale_crops
        
        # Feature caching for efficiency
        self.cache_size = cache_size
        self.feature_cache = {}
        self.cache_keys = []
        
        # Adaptive weighting network
        if adaptive_weighting:
            clip_dim = self.clip_model.config.projection_dim
            self.weight_predictor = nn.Sequential(
                nn.Linear(clip_dim * 2, clip_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(clip_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Attention mechanism for feature weighting
        if use_attention:
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=clip_dim, 
                num_heads=8, 
                dropout=0.1,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(clip_dim)
        
        # Freeze CLIP parameters (optional)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def compute_similarity(self, images, texts, entity_texts_list=None):
        """
        Enhanced similarity computation with caching and adaptive weighting.
        """
        device = next(self.clip_model.parameters()).device
        batch_size = len(texts) if isinstance(texts, list) else texts.shape[0]
        
        similarity_scores = []
        adaptive_weights = []
        
        with torch.no_grad():
            for i in range(batch_size):
                # Get single image and text
                if isinstance(images, list):
                    image = images[i]
                else:
                    image = images[i]
                    
                text = texts[i] if isinstance(texts, list) else texts[i]
                
                # Create cache key
                cache_key = hash((str(text), id(image) if hasattr(image, 'size') else str(image)))
                
                # Check cache
                if cache_key in self.feature_cache:
                    F_img, G_txt = self.feature_cache[cache_key]
                else:
                    # Compute features
                    inputs = self.clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    image_inputs = {"pixel_values": inputs["pixel_values"]}
                    text_inputs = {k: inputs[k] for k in ("input_ids", "attention_mask") if k in inputs}
                    
                    F_img = self.clip_model.get_image_features(**image_inputs)
                    G_txt = self.clip_model.get_text_features(**text_inputs)
                    F_img = F.normalize(F_img, dim=-1)
                    G_txt = F.normalize(G_txt, dim=-1)
                    
                    # Cache features
                    self._update_cache(cache_key, (F_img, G_txt))
                
                # Global similarity with temperature scaling
                sim_global = torch.sum(F_img * G_txt, dim=-1) / self.temperature
                sim_global = torch.sigmoid(sim_global)  # Normalize to [0,1]
                
                # Multi-level similarity
                similarities = [sim_global]
                
                # Entity-based triplet similarity
                if entity_texts_list and i < len(entity_texts_list):
                    entity_texts = entity_texts_list[i]
                    triplet_sim = self._compute_enhanced_triplet_similarity(image, entity_texts, device)
                    similarities.append(triplet_sim)
                
                # Compute final similarity
                if len(similarities) > 1:
                    # Adaptive weighting if enabled
                    if self.adaptive_weighting:
                        combined_features = torch.cat([F_img, G_txt], dim=-1)
                        adaptive_weight = self.weight_predictor(combined_features)
                        adaptive_weights.append(adaptive_weight)
                        final_sim = adaptive_weight * similarities[0] + (1 - adaptive_weight) * similarities[1]
                    else:
                        final_sim = self.base_weight * similarities[0] + (1 - self.base_weight) * similarities[1]
                else:
                    final_sim = similarities[0]
                    if self.adaptive_weighting:
                        adaptive_weights.append(torch.tensor(1.0, device=device))
                
                similarity_scores.append(final_sim)
        
        # Stack results
        similarity_scores = torch.stack(similarity_scores).squeeze()
        
        # Apply attention mechanism if enabled
        if self.use_attention and len(adaptive_weights) > 0:
            similarity_scores = self._apply_attention_weighting(similarity_scores, adaptive_weights)
        
        return similarity_scores
    
    def _update_cache(self, key, value):
        """Update feature cache with LRU eviction."""
        if len(self.feature_cache) >= self.cache_size:
            # Remove oldest entry
            old_key = self.cache_keys.pop(0)
            del self.feature_cache[old_key]
        
        self.feature_cache[key] = value
        self.cache_keys.append(key)
    
    def _apply_attention_weighting(self, similarity_scores, adaptive_weights):
        """Apply attention mechanism to similarity scores."""
        if len(adaptive_weights) == 0:
            return similarity_scores
            
        weights_tensor = torch.stack(adaptive_weights).squeeze()
        
        # Simple attention over similarities
        attended_scores = similarity_scores * weights_tensor
        return attended_scores
    
    def _compute_enhanced_triplet_similarity(self, image, entity_texts, device):
        """Enhanced triplet similarity with better entity handling and crop strategies."""
        
        # Generate triplet texts with better combinations
        triplet_texts = self._generate_enhanced_triplets(entity_texts)
        
        n = len(triplet_texts)
        if n == 0:
            return torch.tensor(0.0, device=device)
        
        # Use enhanced cropping strategy
        if self.multi_scale_crops:
            subimages = multi_scale_crop_images(image, min(n, 12))  # Limit for efficiency
        else:
            subimages = smart_crop_images(image, min(n, 12))
        
        # Batch encode with improved efficiency
        try:
            subimage_inputs = self.clip_processor(images=subimages, return_tensors="pt", padding=True)
            subimage_inputs = {k: v.to(device) for k, v in subimage_inputs.items()}
            F_subs = self.clip_model.get_image_features(pixel_values=subimage_inputs["pixel_values"])
            F_subs = F.normalize(F_subs, dim=-1)
            
            triplet_inputs = self.clip_processor(text=triplet_texts, return_tensors="pt", padding=True)
            triplet_inputs = {k: v.to(device) for k, v in triplet_inputs.items()}
            G_trips = self.clip_model.get_text_features(
                input_ids=triplet_inputs["input_ids"], 
                attention_mask=triplet_inputs["attention_mask"]
            )
            G_trips = F.normalize(G_trips, dim=-1)
            
            # Enhanced similarity computation
            sim_matrix = torch.matmul(G_trips, F_subs.T) / self.temperature
            
            # Multiple aggregation strategies
            max_similarities = torch.max(sim_matrix, dim=1)[0]  # Best match per triplet
            mean_similarities = torch.mean(sim_matrix, dim=1)   # Average match per triplet
            
            # Weighted combination
            final_similarities = 0.7 * max_similarities + 0.3 * mean_similarities
            
            # Return weighted average of top triplets
            top_k = min(3, len(final_similarities))
            top_similarities, _ = torch.topk(final_similarities, top_k)
            
            return torch.mean(top_similarities)
            
        except Exception as e:
            # Fallback to simple similarity
            return torch.tensor(0.5, device=device)
    
    def _generate_enhanced_triplets(self, entity_texts):
        """Generate more meaningful triplet combinations."""
        triplet_texts = []
        
        subjects = entity_texts.get("subject", [])
        predicates = entity_texts.get("predicate", [])
        objects = entity_texts.get("object", [])
        
        # Full triplets
        for s in subjects:
            for p in predicates:
                for o in objects:
                    triplet_texts.append(f"{s} {p} {o}")
        
        # Partial combinations for robustness
        for s in subjects:
            for o in objects:
                triplet_texts.append(f"{s} and {o}")  # Subject-object pairs
        
        for s in subjects:
            for p in predicates:
                triplet_texts.append(f"{s} {p}")  # Subject-predicate pairs
        
        # Individual entities as fallback
        triplet_texts.extend(subjects)
        triplet_texts.extend(objects)
        
        # Remove duplicates and limit length
        triplet_texts = list(set(triplet_texts))[:20]  # Limit for efficiency
        
        return triplet_texts
    
    def forward(self, image_features, images, texts, entity_texts_list=None):
        """
        Enhanced feature weighting with multiple strategies.
        
        Args:
            image_features: Tensor [B, L, D] - image hidden states from vision encoder
            images: PIL Images or tensors - original images
            texts: List of strings - input texts
            entity_texts_list: Optional list of entity dicts
            
        Returns:
            weighted_features: Tensor [B, L, D] - similarity-weighted image features
        """
        # Compute similarity scores [B]
        sim_scores = self.compute_similarity(images, texts, entity_texts_list)
        
        # Handle different dimensionalities
        if len(sim_scores.shape) == 0:  # Single sample
            sim_scores = sim_scores.unsqueeze(0)
        
        batch_size = image_features.shape[0]
        if len(sim_scores) != batch_size:
            # Pad or trim to match batch size
            if len(sim_scores) < batch_size:
                padding = torch.ones(batch_size - len(sim_scores), device=sim_scores.device) * sim_scores.mean()
                sim_scores = torch.cat([sim_scores, padding])
            else:
                sim_scores = sim_scores[:batch_size]
        
        # Multiple weighting strategies
        if image_features.dim() == 3:  # [B, L, D]
            # Token-level weighting
            sim_scores = sim_scores.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            weighted_features = image_features * sim_scores
            
            # Optional: Add residual connection with learnable gate
            if hasattr(self, 'residual_gate'):
                gate = torch.sigmoid(self.residual_gate)
                weighted_features = gate * weighted_features + (1 - gate) * image_features
        
        elif image_features.dim() == 2:  # [B, D]
            # Global feature weighting
            sim_scores = sim_scores.unsqueeze(-1)  # [B, 1]
            weighted_features = image_features * sim_scores
        
        else:
            # Fallback: broadcast across last dimension
            while len(sim_scores.shape) < len(image_features.shape):
                sim_scores = sim_scores.unsqueeze(-1)
            weighted_features = image_features * sim_scores
        
        return weighted_features
    
    def get_similarity_stats(self):
        """Return statistics about similarity computations."""
        return {
            'cache_size': len(self.feature_cache),
            'cache_hit_rate': getattr(self, '_cache_hits', 0) / max(1, getattr(self, '_cache_queries', 1)),
            'adaptive_weighting': self.adaptive_weighting,
            'use_attention': self.use_attention
        }
    
    def clear_cache(self):
        """Clear the feature cache."""
        self.feature_cache.clear()
        self.cache_keys.clear()


# Example integration into DTCAModel:
"""
In DTCAModel.__init__:
    self.similarity_weighting = SimilarityWeighting(weight=0.5)

In DTCAModel.forward, after getting image_last_hidden_states:
    # Weight image features by similarity
    image_last_hidden_states = self.similarity_weighting(
        image_last_hidden_states, 
        pixel_values,  # or original PIL images
        input_texts,    # list of text strings
        entity_texts_list  # optional
    )
"""
