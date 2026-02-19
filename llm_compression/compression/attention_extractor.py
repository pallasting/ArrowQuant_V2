
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

@dataclass
class KeyInformation:
    """Structure to hold extracted key information from text."""
    key_tokens: List[str]
    token_scores: List[float]
    attention_map: Optional[np.ndarray] = None
    entities: Optional[List[str]] = None
    concepts: Optional[List[str]] = None
    relations: Optional[List[Tuple[str, str, float]]] = None

class AttentionBasedExtractor:
    """
    Extracts key tokens and concepts using Transformer attention weights.
    Implementation of 'Attention-Driven Key Information Extraction' from spec.
    """
    
    def __init__(self, arrow_engine):
        self.engine = arrow_engine
        
    def extract_key_information(
        self, 
        text: str,
        top_k: int = 5,
        percentile: float = 80.0,
        include_attention_map: bool = False
    ) -> KeyInformation:
        """
        Extract key tokens based on aggregate attention scores.
        
        Args:
            text: Input text
            top_k: Number of top tokens to return if percentile selection yields too many/few
            percentile: Percentile threshold for selection (0-100)
            include_attention_map: Whether to include the full attention map in output
            
        Returns:
            KeyInformation object
        """
        # 1. Get tokens and attention
        # We need input_ids to map back to tokens
        encoded = self.engine.tokenizer.encode([text])
        input_ids = encoded['input_ids'][0] # numpy array or tensor, take first batch
        
        # Convert to tokens
        tokens = self.engine.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Get attention weights from model
        # Returns tuple of (embeddings, attentions)
        _, attentions = self.engine.encode(text, output_attentions=True)
        
        # attentions is [tuple(layer_atts)], take first batch item
        layer_attentions = attentions[0] 
        
        # 2. Aggregate Attention
        # Method: Average attention across all heads in the last layer 
        # (or last few layers) to get token importance.
        # Attention shape: (1, heads, seq, seq)
        
        # Use last layer attention
        last_layer_attn = layer_attentions[-1] # Shape: (1, heads, seq, seq)
        
        # Squeeze batch dimension if present (it is (1, ...))
        if last_layer_attn.dim() == 4:
            last_layer_attn = last_layer_attn.squeeze(0) # (heads, seq, seq)
            
        # Average over heads -> (seq, seq)
        avg_attn = last_layer_attn.mean(dim=0)
        
        # Calculate token importance scores
        # Metric: How much attention does each token RECEIVE from other tokens?
        # Sum over rows (dim=0) -> importance
        # Note: In BERT attention A[i, j] is attention FROM i TO j.
        # So summing over i (dim=0) gives total attention received by j.
        # We usually exclude [CLS] (index 0) and [SEP] (last) from receiving, 
        # or we look at attention FROM [CLS] to others as a measure of sentence-level importance.
        
        # Approach 1: Attention from [CLS] to tokens (Sequence classification view)
        cls_attn = avg_attn[0, :] # Attention from [CLS] to all tokens
        
        # Approach 2: PageRank / Rollout (Global importance)
        # For simplicity and speed in PoC, let's use CLS attention + Receipt attention
        
        # Let's use Receipt attention (sum over queries)
        # But we remove self-attention diagonal to avoid bias? BERT usually relies on self-attention.
        # Let's keep it simple: Total received attention.
        
        token_scores = avg_attn.sum(dim=0) # (seq,)
        
        # Normalize
        token_scores = token_scores / token_scores.sum()
        token_scores = token_scores.cpu().numpy()
        
        # 3. Select Key Tokens
        # Filter out special tokens
        valid_indices = []
        clean_tokens = []
        clean_scores = []
        
        for idx, token in enumerate(tokens):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            # Remove subword prefix '##' for display
            display_token = token.replace("##", "")
            if not display_token.strip():
                continue
                
            clean_tokens.append(display_token)
            clean_scores.append(token_scores[idx])
            valid_indices.append(idx)
            
        clean_scores = np.array(clean_scores)
        
        if len(clean_scores) == 0:
            return KeyInformation([], [], None)
            
        # Determine threshold
        threshold = np.percentile(clean_scores, percentile)
        
        # Select indices where score >= threshold
        selected_indices = np.where(clean_scores >= threshold)[0]
        
        # Sort by score descending
        sorted_indices = sorted(selected_indices, key=lambda i: clean_scores[i], reverse=True)
        
        # Limit to top_k if needed, or ensure at least some
        if len(sorted_indices) == 0:
             # Fallback to top k
             sorted_indices = np.argsort(clean_scores)[-top_k:][::-1]
             
        final_key_tokens = [clean_tokens[i] for i in sorted_indices]
        final_scores = [float(clean_scores[i]) for i in sorted_indices]
        
        # 4. Extract Relations (New: Advanced Relation Extraction)
        # Map back to original indices to look up in attention map
        original_indices = [valid_indices[i] for i in sorted_indices]
        relations = self._extract_relations_from_map(final_key_tokens, original_indices, avg_attn)
        
        return KeyInformation(
            key_tokens=final_key_tokens,
            token_scores=final_scores,
            attention_map=avg_attn.cpu().numpy() if include_attention_map else None,
            entities=None,
            concepts=final_key_tokens, # Key tokens are base concepts
            relations=relations
        )

    def _extract_relations_from_map(
        self, 
        tokens: List[str], 
        indices: List[int], 
        attn_map: torch.Tensor,
        rel_threshold: float = 0.01
    ) -> List[Tuple[str, str, float]]:
        """
        Extract directional relations between concepts based on attention flow.
        """
        relations = []
        n = len(tokens)
        if n < 2: return []
        
        # Ensure focus on tokens in the sequence
        for i in range(n):
            if tokens[i] in [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}"]:
                continue
            for j in range(n):
                if i == j or tokens[j] in [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}"]:
                    continue
                
                # Attention from token i to token j: A[idx_i, idx_j]
                # In BERT, rows are queries (attending FROM), columns are keys (attending TO)
                # So weight at [idx_i, idx_j] is how much token_i depends on token_j
                weight = attn_map[indices[i], indices[j]].item()
                
                if weight > rel_threshold:
                    # directional: "token_i" -> "token_j" with weight
                    relations.append((tokens[i], tokens[j], weight))
                    
        # Sort by weight
        relations.sort(key=lambda x: x[2], reverse=True)
        return relations
    def extract_key_information_batch(
        self,
        texts: List[str],
        top_k: int = 5,
        percentile: float = 80.0,
        attentions: Optional[List[Tuple[torch.Tensor, ...]]] = None
    ) -> List[KeyInformation]:
        """
        Extract key tokens for multiple texts efficiently.
        """
        if not texts: return []
        
        # 1. Forward pass only if attentions not provided
        if attentions is None:
            _, attentions = self.engine.encode(texts, output_attentions=True)
            
        all_key_info = []
        for i, text in enumerate(texts):
            # ArrowEngine returns List[Tuple[Tensors]]
            # Find batch and item
            batch_idx = i // self.engine.max_batch_size
            item_in_batch = i % self.engine.max_batch_size
            
            if batch_idx >= len(attentions):
                logger.error(f"Batch index {batch_idx} out of range for attentions of length {len(attentions)}")
                all_key_info.append(KeyInformation([], []))
                continue
                
            layer_attentions = attentions[batch_idx] 
            item_attentions = tuple(layer[item_in_batch:item_in_batch+1] for layer in layer_attentions)
            
            # Get tokens
            tokens = self.engine.tokenizer.convert_ids_to_tokens(self.engine.tokenizer.encode([text])['input_ids'][0])

            
            # Last layer, averaged heads
            last_layer_attn = item_attentions[-1].squeeze(0).mean(dim=0) # (seq, seq)
            token_scores = last_layer_attn.sum(dim=0)
            token_scores = (token_scores / token_scores.sum()).cpu().numpy()
            
            # Filter and select
            clean_tokens = []
            clean_scores = []
            for idx, token in enumerate(tokens):
                if token in ["[CLS]", "[SEP]", "[PAD]"] or not token.replace("##", "").strip():
                    continue
                clean_tokens.append(token.replace("##", ""))
                clean_scores.append(token_scores[idx])
            
            cs = np.array(clean_scores)
            if len(cs) == 0:
                all_key_info.append(KeyInformation([], []))
                continue
                
            thresh = np.percentile(cs, percentile)
            sel = np.where(cs >= thresh)[0]
            if len(sel) == 0: sel = np.argsort(cs)[-top_k:]
            
            srt = sorted(sel, key=lambda idx: cs[idx], reverse=True)
            
            # Map back to original indices for relationship extraction
            # cs was built from clean_tokens, which were filtered from full tokens
            # We need to find the index in 'tokens' for each item in 'srt'
            
            # Rebuild index map for filtered tokens
            clean_to_orig = []
            for idx, token in enumerate(tokens):
                 if token in ["[CLS]", "[SEP]", "[PAD]"] or not token.replace("##", "").strip():
                    continue
                 clean_to_orig.append(idx)
            
            final_tokens = [clean_tokens[idx] for idx in srt]
            orig_indices = [clean_to_orig[idx] for idx in srt]
            
            # Extract relations using specialized method
            relations = self._extract_relations_from_map(final_tokens, orig_indices, last_layer_attn)
            
            all_key_info.append(KeyInformation(
                key_tokens=final_tokens,
                token_scores=[float(cs[idx]) for idx in srt],
                relations=relations
            ))
            
        return all_key_info
