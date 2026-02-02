"""
CompilerKV API: Prefill-only KV Compression Interface

This module provides a clean API for the Stage1-Stage2-Stage3 KV compression method.
It can be used directly without modifying the model's attention implementation.

Usage:
    from kv_compression.api import compress_kv_prefill_only
    
    k_comp, v_comp, indices = compress_kv_prefill_only(
        attn_weights=attn,      # Attention weights from prefill
        value_cache=V,          # Value states
        token_logprobs=logp,    # Log probabilities (optional)
        budgets=budgets,        # Per-layer budgets
    )
"""

import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class CompilerKVCompressor:
    """
    CompilerKV Compressor: Implements the three-stage KV compression pipeline.
    
    Stage1: Compute token utility u_t = α_t * ρ_t
    Stage2: Head-aware reweighting using W_head[l,h]
    Stage3: Risk-adaptive threshold gating with M_lex LUT + budget fix
    """
    
    def __init__(
        self,
        num_layers: int = 32,
        num_heads: int = 32,
        tables_dir: str = "Base/tables/outputs",
        utility_window: int = 32,
        obs_window: int = 64,
        n_entropy_bins: int = 20,
        n_ppl_bins: int = 4,
        entropy_range: Tuple[float, float] = (2.0, 10.0),
        ppl_range: Tuple[float, float] = (1.0, 100.0),
        strict_budget: bool = True,
    ):
        """
        Initialize the CompilerKV compressor.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            tables_dir: Directory containing W_head and M_lex tables
            utility_window: Window size for Stage1 utility computation
            obs_window: Observation window size for Stage3 risk signals
            n_entropy_bins: Number of entropy bins for discretization
            n_ppl_bins: Number of PPL bins for discretization
            entropy_range: (min, max) entropy values for binning
            ppl_range: (min, max) PPL values for binning
            strict_budget: If True, pad to exact budget when candidates < budget
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.utility_window = utility_window
        self.obs_window = obs_window
        self.n_entropy_bins = n_entropy_bins
        self.n_ppl_bins = n_ppl_bins
        self.entropy_range = entropy_range
        self.ppl_range = ppl_range
        self.strict_budget = strict_budget
        
        # Load offline tables
        self.W_head = self._load_table(tables_dir, "W_head.npy", 
                                       default_shape=(num_layers, num_heads))
        self.M_lex = self._load_table(tables_dir, "M_lex.npy",
                                      default_shape=(num_layers, n_entropy_bins, n_ppl_bins),
                                      default_value=0.9)
    
    def _load_table(self, tables_dir: str, filename: str, 
                    default_shape: tuple, default_value: float = 1.0) -> np.ndarray:
        """Load a table from file, or create default if not found."""
        possible_paths = [
            os.path.join(tables_dir, filename),
            os.path.join("Base/tables/outputs", filename),
            os.path.join("tables/outputs", filename),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                table = np.load(path)
                print(f"[CompilerKV] Loaded {filename} from {path}, shape={table.shape}")
                return table
        
        print(f"[CompilerKV] Warning: {filename} not found, using default values")
        return np.ones(default_shape, dtype=np.float32) * default_value
    
    def _discretize_entropy(self, entropy: float) -> int:
        """Map entropy to bin index."""
        e_min, e_max = self.entropy_range
        entropy = max(e_min, min(e_max, entropy))
        normalized = (entropy - e_min) / (e_max - e_min)
        return min(int(normalized * (self.n_entropy_bins - 1)), self.n_entropy_bins - 1)
    
    def _discretize_ppl(self, ppl: float) -> int:
        """Map PPL to bin index using log scale."""
        p_min, p_max = self.ppl_range
        log_ppl = math.log(max(ppl, 1e-6))
        log_min, log_max = math.log(p_min), math.log(p_max)
        log_ppl = max(log_min, min(log_max, log_ppl))
        normalized = (log_ppl - log_min) / (log_max - log_min)
        return min(int(normalized * (self.n_ppl_bins - 1)), self.n_ppl_bins - 1)
    
    def compute_stage1_utility(
        self,
        attn_weights: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stage1: Compute token utility u_t = α_t * ρ_t
        
        Args:
            attn_weights: [L, bsz, H, q_len, kv_len] or [bsz, H, q_len, kv_len]
            value_states: [L, bsz, H, kv_len, d] or [bsz, H, kv_len, d]
        
        Returns:
            u_t: Token utility scores [bsz, kv_len]
        """
        # Handle different input shapes
        if attn_weights.dim() == 5:
            # [L, bsz, H, q_len, kv_len] -> average across layers
            attn_weights = attn_weights.mean(dim=0)
        if value_states.dim() == 5:
            value_states = value_states.mean(dim=0)
        
        bsz, num_heads, q_len, kv_len = attn_weights.shape
        device = attn_weights.device
        dtype = attn_weights.dtype
        
        # α_t: Windowed cumulative attention
        mean_attn = attn_weights.mean(dim=1)  # [bsz, q_len, kv_len]
        
        alpha = torch.zeros(bsz, kv_len, device=device, dtype=dtype)
        w = self.utility_window
        
        for t in range(kv_len):
            j_start = t
            j_end = min(t + w, q_len)
            if j_start < j_end:
                alpha[:, t] = mean_attn[:, j_start:j_end, t].sum(dim=1)
        
        # ρ_t: Relative value norm
        mean_value = value_states.mean(dim=1)  # [bsz, kv_len, head_dim]
        value_norms = torch.norm(mean_value, p=2, dim=-1)  # [bsz, kv_len]
        mean_norm = value_norms.mean(dim=-1, keepdim=True) + 1e-8
        rho = value_norms / mean_norm
        
        # u_t = α_t * ρ_t
        return alpha * rho
    
    def compute_stage2_reweight(
        self,
        u_t: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Stage2: Apply head-aware reweighting.
        
        Args:
            u_t: Token utility [bsz, kv_len]
            layer_idx: Current layer index
        
        Returns:
            u_hat: Reweighted utility [bsz, kv_len]
        """
        if layer_idx < self.W_head.shape[0]:
            max_weight = float(np.max(self.W_head[layer_idx, :]))
        else:
            max_weight = 1.0
        
        return u_t * max_weight
    
    def compute_risk_signals(
        self,
        attn_weights: torch.Tensor,
        token_logprobs: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """
        Compute risk signals: attention entropy and PPL.
        
        Args:
            attn_weights: [bsz, H, q_len, kv_len] or [L, bsz, H, q_len, kv_len]
            token_logprobs: [bsz, q_len] log probabilities
        
        Returns:
            entropy: Attention entropy H(A)
            ppl: Local perplexity
        """
        if attn_weights.dim() == 5:
            attn_weights = attn_weights.mean(dim=0)
        
        bsz, num_heads, q_len, kv_len = attn_weights.shape
        
        # Observation window
        w_obs = min(self.obs_window, q_len)
        omega_start = max(0, q_len - w_obs)
        
        # Average attention distribution
        omega_attn = attn_weights[:, :, omega_start:, :]
        A_tilde = omega_attn.mean(dim=(1, 2))  # [bsz, kv_len]
        A_tilde = A_tilde / (A_tilde.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Attention entropy
        entropy_batch = -torch.sum(A_tilde * torch.log(A_tilde + 1e-8), dim=-1)
        entropy = entropy_batch.mean().item()
        
        # PPL
        if token_logprobs is not None:
            omega_logprobs = token_logprobs[:, omega_start:]
            mean_nll = -omega_logprobs.mean(dim=-1)
            ppl = torch.exp(mean_nll).mean().item()
        else:
            ppl = 10.0  # Default
        
        return entropy, ppl
    
    def compute_stage3_selection(
        self,
        u_hat: torch.Tensor,
        layer_idx: int,
        entropy: float,
        ppl: float,
        budget: int,
    ) -> torch.Tensor:
        """
        Stage3: Risk-adaptive threshold gating + budget fix.
        
        Args:
            u_hat: Reweighted utility [bsz, kv_len]
            layer_idx: Current layer index
            entropy: Attention entropy
            ppl: Local perplexity
            budget: Number of tokens to keep
        
        Returns:
            keep_mask: Boolean mask [bsz, kv_len]
        """
        bsz, kv_len = u_hat.shape
        device = u_hat.device
        
        # Discretize risk signals
        b_h = self._discretize_entropy(entropy)
        b_p = self._discretize_ppl(ppl)
        
        # Look up threshold
        if layer_idx < self.M_lex.shape[0]:
            tau = self.M_lex[layer_idx, b_h, b_p]
        else:
            tau = 0.9
        
        # Normalize scores for thresholding
        u_hat_norm = u_hat / (u_hat.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        # Create keep mask
        keep_mask = torch.zeros(bsz, kv_len, dtype=torch.bool, device=device)
        
        for b in range(bsz):
            scores = u_hat[b]
            scores_norm = u_hat_norm[b]
            
            # Threshold gating
            cand_mask = scores_norm >= tau
            cand_indices = torch.where(cand_mask)[0]
            num_cand = len(cand_indices)
            
            if num_cand > budget:
                # Keep top-budget
                top_k = scores.topk(budget, largest=True).indices
                keep_mask[b, top_k] = True
            elif num_cand < budget and self.strict_budget:
                # Include all candidates + fill from remaining
                keep_mask[b, cand_indices] = True
                non_cand_mask = ~cand_mask
                non_cand_indices = torch.where(non_cand_mask)[0]
                
                if len(non_cand_indices) > 0:
                    num_to_add = min(budget - num_cand, len(non_cand_indices))
                    extra_scores = scores[non_cand_indices]
                    top_extra = extra_scores.topk(num_to_add, largest=True).indices
                    keep_mask[b, non_cand_indices[top_extra]] = True
            else:
                keep_mask[b, cand_indices] = True
        
        return keep_mask
    
    def compress(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attn_weights: torch.Tensor,
        budgets: Union[int, List[int]],
        token_logprobs: Optional[torch.Tensor] = None,
        window_size: int = 64,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Full compression pipeline: Stage1 + Stage2 + Stage3.
        
        Args:
            key_states: [L, bsz, H, kv_len, d] Key states per layer
            value_states: [L, bsz, H, kv_len, d] Value states per layer
            attn_weights: [L, bsz, H, q_len, kv_len] Attention weights per layer
            budgets: Per-layer budget or single budget for all layers
            token_logprobs: [bsz, q_len] Optional log probabilities
            window_size: Recent window to always keep
        
        Returns:
            k_compressed: List of compressed key states per layer
            v_compressed: List of compressed value states per layer
            keep_indices: List of kept indices per layer
        """
        num_layers = key_states.shape[0]
        bsz, num_heads, kv_len, head_dim = key_states[0].shape
        
        # Convert budgets to list
        if isinstance(budgets, int):
            budgets = [budgets] * num_layers
        
        # Stage1: Compute global token utility
        u_t = self.compute_stage1_utility(attn_weights, value_states)
        
        # Compute risk signals once (shared across layers)
        entropy, ppl = self.compute_risk_signals(attn_weights, token_logprobs)
        
        k_compressed = []
        v_compressed = []
        keep_indices = []
        
        for l in range(num_layers):
            K_l = key_states[l]
            V_l = value_states[l]
            B_l = budgets[l]
            
            # Stage2: Reweight
            u_hat_l = self.compute_stage2_reweight(u_t, l)
            
            # Split into past tokens (to compress) and window (always keep)
            if kv_len > window_size:
                u_hat_past = u_hat_l[:, :-window_size]
                K_past = K_l[:, :, :-window_size, :]
                V_past = V_l[:, :, :-window_size, :]
                K_window = K_l[:, :, -window_size:, :]
                V_window = V_l[:, :, -window_size:, :]
                
                past_budget = min(B_l - window_size, kv_len - window_size)
                
                # Stage3: Select tokens
                keep_mask = self.compute_stage3_selection(
                    u_hat_past, l, entropy, ppl, past_budget
                )
                
                # Gather compressed KV
                keep_idx_list = []
                K_comp_list = []
                V_comp_list = []
                
                for b in range(bsz):
                    idx = torch.where(keep_mask[b])[0]
                    keep_idx_list.append(idx)
                    
                    idx_expanded = idx.unsqueeze(0).unsqueeze(-1).expand(num_heads, -1, head_dim)
                    K_comp_list.append(K_past[b].gather(dim=1, index=idx_expanded))
                    V_comp_list.append(V_past[b].gather(dim=1, index=idx_expanded))
                
                # Pad and stack for batch
                max_keep = max(len(idx) for idx in keep_idx_list)
                K_comp = torch.zeros(bsz, num_heads, max_keep, head_dim, 
                                     device=K_l.device, dtype=K_l.dtype)
                V_comp = torch.zeros(bsz, num_heads, max_keep, head_dim,
                                     device=V_l.device, dtype=V_l.dtype)
                
                for b in range(bsz):
                    n = K_comp_list[b].shape[1]
                    K_comp[b, :, :n, :] = K_comp_list[b]
                    V_comp[b, :, :n, :] = V_comp_list[b]
                
                # Concatenate with window
                K_final = torch.cat([K_comp, K_window], dim=2)
                V_final = torch.cat([V_comp, V_window], dim=2)
            else:
                # Sequence too short, keep all
                K_final = K_l
                V_final = V_l
                keep_idx_list = [torch.arange(kv_len, device=K_l.device) for _ in range(bsz)]
            
            k_compressed.append(K_final)
            v_compressed.append(V_final)
            keep_indices.append(keep_idx_list)
        
        return k_compressed, v_compressed, keep_indices


def compress_kv_prefill_only(
    attn_weights: torch.Tensor,
    value_cache: torch.Tensor,
    budgets: Union[int, List[int], Dict[int, int]],
    token_logprobs: Optional[torch.Tensor] = None,
    key_cache: Optional[torch.Tensor] = None,
    tables_dir: str = "Base/tables/outputs",
    w: int = 32,
    w_obs: int = 64,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List]:
    """
    Compress KV cache during prefill using CompilerKV method.
    
    This is the main API function as specified in the design document.
    
    Args:
        attn_weights: Attention weights [L, bsz, H, q_len, kv_len] or stats for Ω
        value_cache: Value states [L, bsz, H, kv_len, d]
        budgets: Per-layer budgets {layer_idx: budget} or single int
        token_logprobs: Log p(x_j | x_<j) for j in Ω [bsz, |Ω|]
        key_cache: Key states [L, bsz, H, kv_len, d] (if not provided, use value shapes)
        tables_dir: Path to W_head and M_lex tables
        w: Utility window size
        w_obs: Observation window size
    
    Returns:
        K_comp_list: Compressed key states per layer
        V_comp_list: Compressed value states per layer
        keep_indices_list: Indices of kept tokens per layer
    """
    # Get dimensions
    num_layers = value_cache.shape[0]
    bsz, num_heads, kv_len, head_dim = value_cache[0].shape
    
    # Create compressor
    compressor = CompilerKVCompressor(
        num_layers=num_layers,
        num_heads=num_heads,
        tables_dir=tables_dir,
        utility_window=w,
        obs_window=w_obs,
    )
    
    # If key_cache not provided, assume same as value_cache (for utility computation only)
    if key_cache is None:
        key_cache = value_cache
    
    # Convert budgets to list
    if isinstance(budgets, int):
        budgets_list = [budgets] * num_layers
    elif isinstance(budgets, dict):
        budgets_list = [budgets.get(l, budgets.get(0, 2048)) for l in range(num_layers)]
    else:
        budgets_list = list(budgets)
    
    # Run compression
    K_comp_list, V_comp_list, keep_indices_list = compressor.compress(
        key_states=key_cache,
        value_states=value_cache,
        attn_weights=attn_weights,
        budgets=budgets_list,
        token_logprobs=token_logprobs,
        window_size=w_obs,
    )
    
    return K_comp_list, V_comp_list, keep_indices_list


# Convenience alias
KVCompressor = CompilerKVCompressor
