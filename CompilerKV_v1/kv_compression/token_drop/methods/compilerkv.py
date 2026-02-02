"""
CompilerKV: Prefill-only KV Compression (Stage1 + Stage2 + Stage3)

Based on the specification document:
- Stage1: Token Utility u_t (layer-agnostic common score)
- Stage2: Head-aware reweighting using W_head[l,h]
- Stage3: Risk-adaptive threshold gating + budget fix using M_lex LUT
"""

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Tuple, Dict, List


class CompilerKV:
    """
    CompilerKV: Stage1-Stage2-Stage3 Prefill-only KV Compression.
    
    Stage1: Compute token utility u_t = α_t * ρ_t
    Stage2: Head-aware reweighting using W_head table
    Stage3: Risk-adaptive threshold gating with M_lex LUT + budget fix
    """
    
    def __init__(
        self,
        num_hidden_layers: int = 32,
        num_heads: int = 32,
        window_size: int = 64,  # observation window size (w_obs)
        utility_window: int = 32,  # utility window (w) for Stage1
        max_capacity_prompt: int = 2048,
        layer_idx: int = None,
        tables_dir: str = "Base/tables/outputs",
        # Entropy and PPL binning parameters
        n_entropy_bins: int = 20,
        n_ppl_bins: int = 4,
        entropy_range: Tuple[float, float] = (2.0, 10.0),
        ppl_range: Tuple[float, float] = (1.0, 100.0),
        # Budget strategy
        strict_budget: bool = True,  # if True, pad to B_l when I_cand < B_l
    ):
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        
        self.window_size = window_size  # observation window for risk signals
        self.utility_window = utility_window  # window for utility computation
        self.max_capacity_prompt = max_capacity_prompt
        self.base_budget = self.max_capacity_prompt - self.window_size
        
        self.n_entropy_bins = n_entropy_bins
        self.n_ppl_bins = n_ppl_bins
        self.entropy_range = entropy_range
        self.ppl_range = ppl_range
        self.strict_budget = strict_budget
        
        # Load offline tables
        self.tables_dir = tables_dir
        self.W_head = None  # [L, H]
        self.M_lex = None   # [L, n_entropy_bins, n_ppl_bins]
        self._load_tables()
    
    def _load_tables(self):
        """Load W_head and M_lex tables from files."""
        # Try multiple possible paths
        possible_dirs = [
            self.tables_dir,
            os.path.join(os.path.dirname(__file__), "../../../../tables/outputs"),
            os.path.join(os.path.dirname(__file__), "../../../tables/outputs"),
            "Base/tables/outputs",
            "tables/outputs",
        ]
        
        W_head_loaded = False
        M_lex_loaded = False
        
        for table_dir in possible_dirs:
            if not os.path.exists(table_dir):
                continue
            
            W_head_path = os.path.join(table_dir, "W_head.npy")
            M_lex_path = os.path.join(table_dir, "M_lex.npy")
            
            if os.path.exists(W_head_path) and not W_head_loaded:
                self.W_head = np.load(W_head_path)
                W_head_loaded = True
                if self.layer_idx == 0:
                    print(f"[CompilerKV] Loaded W_head from {W_head_path}, shape={self.W_head.shape}")
            
            if os.path.exists(M_lex_path) and not M_lex_loaded:
                self.M_lex = np.load(M_lex_path)
                M_lex_loaded = True
                if self.layer_idx == 0:
                    print(f"[CompilerKV] Loaded M_lex from {M_lex_path}, shape={self.M_lex.shape}")
        
        # If tables not found, create default ones
        if self.W_head is None:
            if self.layer_idx == 0:
                print("[CompilerKV] Warning: W_head table not found, using default ones")
            self.W_head = np.ones((self.num_hidden_layers, self.num_heads), dtype=np.float32)
        
        if self.M_lex is None:
            if self.layer_idx == 0:
                print("[CompilerKV] Warning: M_lex table not found, using default threshold 0.9")
            self.M_lex = np.ones((self.num_hidden_layers, self.n_entropy_bins, self.n_ppl_bins), dtype=np.float32) * 0.9

    def _discretize_entropy(self, entropy: float) -> int:
        """Discretize entropy value to bin index."""
        e_min, e_max = self.entropy_range
        # Clamp to range
        entropy = max(e_min, min(e_max, entropy))
        # Normalize to [0, 1]
        normalized = (entropy - e_min) / (e_max - e_min)
        # Map to bin index
        bin_idx = int(normalized * (self.n_entropy_bins - 1))
        return min(bin_idx, self.n_entropy_bins - 1)
    
    def _discretize_ppl(self, ppl: float) -> int:
        """Discretize PPL value to bin index."""
        p_min, p_max = self.ppl_range
        # Use log scale for PPL
        log_ppl = math.log(max(ppl, 1e-6))
        log_min = math.log(p_min)
        log_max = math.log(p_max)
        # Clamp to range
        log_ppl = max(log_min, min(log_max, log_ppl))
        # Normalize to [0, 1]
        normalized = (log_ppl - log_min) / (log_max - log_min)
        # Map to bin index
        bin_idx = int(normalized * (self.n_ppl_bins - 1))
        return min(bin_idx, self.n_ppl_bins - 1)

    @staticmethod
    def compute_stage1_utility(
        attn_weights: torch.Tensor,  # [bsz, num_heads, q_len, kv_len]
        value_states: torch.Tensor,  # [bsz, num_heads, kv_len, head_dim]
        window: int = 32,
    ) -> torch.Tensor:
        """
        Stage1: Compute token utility u_t = α_t * ρ_t
        
        Args:
            attn_weights: Attention weights after softmax, shape [bsz, num_heads, q_len, kv_len]
            value_states: Value states, shape [bsz, num_heads, kv_len, head_dim]
            window: Window size for utility computation
        
        Returns:
            u_t: Token utility scores, shape [bsz, kv_len]
        """
        bsz, num_heads, q_len, kv_len = attn_weights.shape
        device = attn_weights.device
        dtype = attn_weights.dtype
        
        # === Compute α_t: windowed cumulative attention ===
        # Average attention across heads: Ā_{j,t} = mean_{h} A_{j,t}^{h}
        # Shape: [bsz, q_len, kv_len]
        mean_attn = attn_weights.mean(dim=1)
        
        # For each token t, sum attention from j=t to min(t+w, T)
        # α_t = Σ_{j=t}^{min(t+w, T)} Ā_{j,t}
        alpha = torch.zeros(bsz, kv_len, device=device, dtype=dtype)
        
        for t in range(kv_len):
            j_start = t
            j_end = min(t + window, q_len)
            if j_start < j_end:
                # Sum attention received by token t from positions [j_start, j_end)
                alpha[:, t] = mean_attn[:, j_start:j_end, t].sum(dim=1)
        
        # === Compute ρ_t: relative value norm ===
        # Average value across heads: v̄_t = mean_{h} v_t^{h}
        # Shape: [bsz, kv_len, head_dim]
        mean_value = value_states.mean(dim=1)
        
        # Compute L2 norm of each token's value
        # Shape: [bsz, kv_len]
        value_norms = torch.norm(mean_value, p=2, dim=-1)
        
        # Sample-wise normalization: ρ_t = ||v̄_t||_2 / mean_i ||v̄_i||_2
        mean_norm = value_norms.mean(dim=-1, keepdim=True) + 1e-8
        rho = value_norms / mean_norm
        
        # === Final utility: u_t = α_t * ρ_t ===
        u_t = alpha * rho
        
        return u_t

    @staticmethod
    def compute_stage1_utility_fast(
        attn_weights: torch.Tensor,  # [bsz, num_heads, q_len, kv_len]
        value_states: torch.Tensor,  # [bsz, num_heads, kv_len, head_dim]
        window: int = 32,
    ) -> torch.Tensor:
        """
        Stage1: Compute token utility u_t = α_t * ρ_t (Optimized version)
        Uses cumulative sum for efficient windowed attention computation.
        """
        bsz, num_heads, q_len, kv_len = attn_weights.shape
        device = attn_weights.device
        dtype = attn_weights.dtype
        
        # === Compute α_t: windowed cumulative attention ===
        # Average attention across heads
        mean_attn = attn_weights.mean(dim=1)  # [bsz, q_len, kv_len]
        
        # Create causal mask for window
        # For each key position t, we want attention from query positions [t, min(t+w, q_len)]
        # Use cumsum trick: cumsum along q dimension, then take difference
        
        # Cumulative sum along query dimension
        cumsum_attn = mean_attn.cumsum(dim=1)  # [bsz, q_len, kv_len]
        
        # α_t = cumsum[min(t+w, q_len)] - cumsum[t-1]
        alpha = torch.zeros(bsz, kv_len, device=device, dtype=dtype)
        
        for t in range(kv_len):
            j_end = min(t + window, q_len)
            if j_end > 0:
                upper = cumsum_attn[:, j_end - 1, t]
                lower = cumsum_attn[:, t - 1, t] if t > 0 else 0
                alpha[:, t] = upper - lower
        
        # === Compute ρ_t: relative value norm ===
        mean_value = value_states.mean(dim=1)  # [bsz, kv_len, head_dim]
        value_norms = torch.norm(mean_value, p=2, dim=-1)  # [bsz, kv_len]
        mean_norm = value_norms.mean(dim=-1, keepdim=True) + 1e-8
        rho = value_norms / mean_norm
        
        # === Final utility ===
        u_t = alpha * rho
        
        return u_t

    def stage2_reweight(
        self,
        u_t: torch.Tensor,  # [bsz, kv_len]
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Stage2: Head-aware reweighting using W_head table.
        
        Since u_t is already head-aggregated, we use the max head weight for each layer.
        û_t^(l) = u_t * max_h W_head[l,h]
        
        Args:
            u_t: Token utility from Stage1, shape [bsz, kv_len]
            layer_idx: Current layer index
        
        Returns:
            u_hat: Reweighted utility, shape [bsz, kv_len]
        """
        # Get head weights for this layer
        if layer_idx < self.W_head.shape[0]:
            head_weights = self.W_head[layer_idx, :]  # [num_heads]
            max_weight = np.max(head_weights)
        else:
            max_weight = 1.0
        
        # Apply reweighting
        u_hat = u_t * max_weight
        
        return u_hat

    def stage2_reweight_per_head(
        self,
        u_t: torch.Tensor,  # [bsz, kv_len]
        attn_weights: torch.Tensor,  # [bsz, num_heads, q_len, kv_len]
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Stage2: Per-head reweighting, then aggregate to token level.
        
        û_t^(l,h) = u_t * W_head[l,h]
        û_t^(l) = max_h û_t^(l,h)
        
        Args:
            u_t: Token utility from Stage1, shape [bsz, kv_len]
            attn_weights: Attention weights, shape [bsz, num_heads, q_len, kv_len]
            layer_idx: Current layer index
        
        Returns:
            u_hat: Reweighted utility, shape [bsz, kv_len]
        """
        bsz, num_heads, q_len, kv_len = attn_weights.shape
        device = u_t.device
        dtype = u_t.dtype
        
        # Get head weights for this layer
        if layer_idx < self.W_head.shape[0]:
            head_weights = torch.tensor(
                self.W_head[layer_idx, :num_heads], 
                device=device, 
                dtype=dtype
            )  # [num_heads]
        else:
            head_weights = torch.ones(num_heads, device=device, dtype=dtype)
        
        # Expand u_t to per-head: [bsz, num_heads, kv_len]
        u_t_expanded = u_t.unsqueeze(1).expand(-1, num_heads, -1)
        
        # Apply per-head weights: û_t^(l,h) = u_t * W_head[l,h]
        u_hat_per_head = u_t_expanded * head_weights.view(1, -1, 1)
        
        # Aggregate: û_t^(l) = max_h û_t^(l,h)
        u_hat, _ = u_hat_per_head.max(dim=1)  # [bsz, kv_len]
        
        return u_hat

    def compute_risk_signals(
        self,
        attn_weights: torch.Tensor,  # [bsz, num_heads, q_len, kv_len]
        token_logprobs: Optional[torch.Tensor] = None,  # [bsz, q_len]
    ) -> Tuple[float, float]:
        """
        Compute risk signals for Stage3: attention entropy and PPL.
        
        Args:
            attn_weights: Attention weights, shape [bsz, num_heads, q_len, kv_len]
            token_logprobs: Log probabilities for each token (teacher forcing), shape [bsz, q_len]
        
        Returns:
            entropy: Attention entropy H(A)
            ppl: Local perplexity
        """
        bsz, num_heads, q_len, kv_len = attn_weights.shape
        
        # Define observation window Ω = {T - w_obs + 1, ..., T}
        w_obs = min(self.window_size, q_len)
        omega_start = max(0, q_len - w_obs)
        
        # === Compute average attention distribution Ã_t ===
        # Ã_t = (1/|Ω|LH) Σ_{j∈Ω} Σ_l Σ_h A_{j,t}^{(l,h)}
        # Since we only have one layer here, we compute within this layer
        # and assume aggregation happens at the final stage
        
        # Average over observation window and heads
        omega_attn = attn_weights[:, :, omega_start:, :]  # [bsz, heads, w_obs, kv_len]
        A_tilde = omega_attn.mean(dim=(1, 2))  # [bsz, kv_len]
        
        # Normalize to make it a proper distribution
        A_tilde = A_tilde / (A_tilde.sum(dim=-1, keepdim=True) + 1e-8)
        
        # === Compute attention entropy H(A) ===
        # H(A) = -Σ_t Ã_t log Ã_t
        entropy_per_batch = -torch.sum(
            A_tilde * torch.log(A_tilde + 1e-8), 
            dim=-1
        )  # [bsz]
        entropy = entropy_per_batch.mean().item()
        
        # === Compute local perplexity PPL ===
        if token_logprobs is not None:
            # PPL = exp(-1/|Ω| Σ_{j∈Ω} log p(x_j | x_<j))
            omega_logprobs = token_logprobs[:, omega_start:]  # [bsz, w_obs]
            mean_nll = -omega_logprobs.mean(dim=-1)  # [bsz]
            ppl = torch.exp(mean_nll).mean().item()
        else:
            # If no logprobs provided, use default middle bin
            ppl = 10.0
        
        return entropy, ppl

    def stage3_threshold_gating(
        self,
        u_hat: torch.Tensor,  # [bsz, kv_len]
        layer_idx: int,
        entropy: float,
        ppl: float,
        budget: int,
    ) -> torch.Tensor:
        """
        Stage3: Risk-adaptive threshold gating + budget fix.
        
        Args:
            u_hat: Reweighted utility from Stage2, shape [bsz, kv_len]
            layer_idx: Current layer index
            entropy: Attention entropy
            ppl: Local perplexity
            budget: Budget B_l for this layer
        
        Returns:
            keep_indices: Indices of tokens to keep, shape [bsz, num_keep]
        """
        bsz, kv_len = u_hat.shape
        device = u_hat.device
        
        # === Discretize risk signals ===
        b_h = self._discretize_entropy(entropy)
        b_p = self._discretize_ppl(ppl)
        
        # === Look up threshold from M_lex ===
        if layer_idx < self.M_lex.shape[0]:
            tau = self.M_lex[layer_idx, b_h, b_p]
        else:
            tau = 0.9
        
        # Normalize u_hat to [0, 1] for threshold comparison
        u_hat_norm = u_hat / (u_hat.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        # === Threshold gating: I_cand = {t | û_t >= τ} ===
        mask_cand = u_hat_norm >= tau  # [bsz, kv_len]
        
        # === Budget fix: ensure we keep exactly budget tokens ===
        keep_indices_list = []
        
        for b in range(bsz):
            u_hat_b = u_hat[b]  # [kv_len]
            mask_b = mask_cand[b]  # [kv_len]
            
            # Get candidate indices
            cand_indices = torch.where(mask_b)[0]
            num_cand = len(cand_indices)
            
            if num_cand > budget:
                # Too many candidates: keep top-B_l by û_t
                cand_scores = u_hat_b[cand_indices]
                top_k_idx = cand_scores.topk(budget, largest=True).indices
                keep_idx = cand_indices[top_k_idx]
            elif num_cand < budget and self.strict_budget:
                # Too few candidates: pad from remaining tokens
                non_cand_mask = ~mask_b
                non_cand_indices = torch.where(non_cand_mask)[0]
                
                if len(non_cand_indices) > 0:
                    non_cand_scores = u_hat_b[non_cand_indices]
                    num_to_add = min(budget - num_cand, len(non_cand_indices))
                    top_k_idx = non_cand_scores.topk(num_to_add, largest=True).indices
                    extra_idx = non_cand_indices[top_k_idx]
                    keep_idx = torch.cat([cand_indices, extra_idx])
                else:
                    keep_idx = cand_indices
            else:
                # Keep all candidates (either exact or allow under-budget)
                keep_idx = cand_indices
            
            # Sort indices to maintain order
            keep_idx = keep_idx.sort()[0]
            keep_indices_list.append(keep_idx)
        
        # Pad to same length for batching
        max_keep = max(len(idx) for idx in keep_indices_list)
        keep_indices = torch.zeros(bsz, max_keep, dtype=torch.long, device=device)
        
        for b, idx in enumerate(keep_indices_list):
            keep_indices[b, :len(idx)] = idx
            if len(idx) < max_keep:
                # Pad with last valid index
                keep_indices[b, len(idx):] = idx[-1] if len(idx) > 0 else 0
        
        return keep_indices

    def compress_kv(
        self,
        key_states: torch.Tensor,  # [bsz, num_heads, kv_len, head_dim]
        value_states: torch.Tensor,  # [bsz, num_heads, kv_len, head_dim]
        query_states: torch.Tensor,  # [bsz, num_heads, q_len, head_dim]
        attention_mask: Optional[torch.Tensor] = None,
        token_logprobs: Optional[torch.Tensor] = None,
        layer_idx: int = None,
        budget: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Main compression function: Stage1 + Stage2 + Stage3.
        
        Args:
            key_states: Key states, shape [bsz, num_heads, kv_len, head_dim]
            value_states: Value states, shape [bsz, num_heads, kv_len, head_dim]
            query_states: Query states, shape [bsz, num_heads, q_len, head_dim]
            attention_mask: Optional attention mask
            token_logprobs: Optional log probabilities for PPL computation
            layer_idx: Current layer index
            budget: Budget for this layer (if None, use self.base_budget)
        
        Returns:
            k_compressed: Compressed key states
            v_compressed: Compressed value states
            keep_indices: Indices of kept tokens
        """
        if layer_idx is None:
            layer_idx = self.layer_idx
        if budget is None:
            budget = self.base_budget
        
        bsz, num_heads, kv_len, head_dim = key_states.shape
        
        # Skip compression if sequence is short
        if kv_len <= self.window_size:
            return key_states, value_states, None
        
        # Compute attention weights for utility computation
        # (This is a simplified version - in practice, may reuse prefill attention)
        with torch.no_grad():
            attn_weights = torch.matmul(
                query_states, 
                key_states.transpose(-2, -1)
            ) / math.sqrt(head_dim)
            
            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(kv_len, kv_len, device=key_states.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0),
                float('-inf')
            )
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_weights = attn_weights.to(key_states.dtype)
        
        # === Stage1: Compute token utility ===
        u_t = self.compute_stage1_utility(
            attn_weights, 
            value_states, 
            window=self.utility_window
        )
        
        # === Stage2: Head-aware reweighting ===
        u_hat = self.stage2_reweight_per_head(u_t, attn_weights, layer_idx)
        
        # === Compute risk signals ===
        entropy, ppl = self.compute_risk_signals(attn_weights, token_logprobs)
        
        # === Stage3: Threshold gating + budget fix ===
        # Adjust budget to exclude window (recent tokens always kept)
        compression_budget = min(budget, kv_len - self.window_size)
        
        # Apply gating only to non-window tokens
        u_hat_past = u_hat[:, :-self.window_size]
        keep_indices = self.stage3_threshold_gating(
            u_hat_past, 
            layer_idx, 
            entropy, 
            ppl, 
            compression_budget
        )
        
        # === Gather compressed KV ===
        # Expand indices for gathering
        indices_expanded = keep_indices.unsqueeze(1).unsqueeze(-1)
        indices_expanded = indices_expanded.expand(-1, num_heads, -1, head_dim)
        
        # Gather past tokens
        k_past = key_states[:, :, :-self.window_size, :]
        v_past = value_states[:, :, :-self.window_size, :]
        
        k_compressed_past = k_past.gather(dim=2, index=indices_expanded)
        v_compressed_past = v_past.gather(dim=2, index=indices_expanded)
        
        # Concat with window (recent tokens)
        k_window = key_states[:, :, -self.window_size:, :]
        v_window = value_states[:, :, -self.window_size:, :]
        
        k_compressed = torch.cat([k_compressed_past, k_window], dim=2)
        v_compressed = torch.cat([v_compressed_past, v_window], dim=2)
        
        return k_compressed, v_compressed, keep_indices


class CompilerKVCluster(CompilerKV):
    """
    Wrapper class compatible with existing DynamicKV interface.
    Provides budget_compute_per_layer and update_and_reset_budget methods.
    """
    
    def __init__(
        self,
        num_hidden_layers: int = 32,
        num_heads: int = 32,
        window_size: int = 64,
        max_capacity_prompt: int = 2048,
        kernel_size: int = 7,
        pooling: str = 'avgpool',
        layer_idx: int = None,
        tables_dir: str = "Base/tables/outputs",
        radio_max: float = 10.0,
        radio_min: float = 0.1,
    ):
        super().__init__(
            num_hidden_layers=num_hidden_layers,
            num_heads=num_heads,
            window_size=window_size,
            utility_window=32,
            max_capacity_prompt=max_capacity_prompt,
            layer_idx=layer_idx,
            tables_dir=tables_dir,
        )
        
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.radio_max = radio_max
        self.radio_min = radio_min
        self.budget_size = -1
        
    def budget_compute_per_layer(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Compute per-layer budget and return compressed KV cache.
        Compatible with DynamicKV interface.
        """
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if self.layer_idx == 0:
            print(f"[CompilerKV] max_capacity_prompt: {self.max_capacity_prompt}, "
                  f"window_size: {self.window_size}, base_budget: {self.base_budget}")
        
        if q_len < self.window_size:
            return None
        
        # Compute budget size for this sample (reset for each new sample)
        past_len = q_len - self.window_size
        self.budget_size = min(
            int(self.radio_max * self.base_budget), 
            past_len
        )
        
        # Compute attention weights
        with torch.no_grad():
            attn_weights = torch.matmul(
                query_states[..., -self.window_size:, :], 
                key_states.transpose(2, 3)
            ) / math.sqrt(head_dim)
            
            # Apply causal mask for window
            mask = torch.full(
                (self.window_size, self.window_size), 
                torch.finfo(attn_weights.dtype).min, 
                device=attn_weights.device
            )
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            attn_weights[:, :, -self.window_size:, -self.window_size:] += mask[None, None, :, :]
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_weights = attn_weights.to(query_states.dtype)
        
        # === Stage1: Compute token utility ===
        # Only consider past tokens (exclude window)
        attn_weights_past = attn_weights[:, :, :, :-self.window_size]  # [bsz, heads, window, past_len]
        value_past = value_states[:, :, :-self.window_size, :]  # [bsz, heads, past_len, dim]
        
        # Compute attention sum for each past token (how much attention it receives)
        attn_sum = attn_weights_past.sum(dim=2)  # [bsz, heads, past_len]
        
        # Apply pooling (for smoothing)
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(
                attn_sum, 
                kernel_size=self.kernel_size, 
                padding=self.kernel_size // 2, 
                stride=1
            )
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(
                attn_sum, 
                kernel_size=self.kernel_size, 
                padding=self.kernel_size // 2, 
                stride=1
            )
        else:
            attn_cache = attn_sum
        
        # === Stage2: Apply head weights ===
        if self.layer_idx < self.W_head.shape[0]:
            head_weights = torch.tensor(
                self.W_head[self.layer_idx, :num_heads],
                device=attn_cache.device,
                dtype=attn_cache.dtype
            ).view(1, -1, 1)
            attn_cache = attn_cache * head_weights
        
        # Mean across heads for token-level score
        attn_cache_mean = attn_cache.mean(dim=1)  # [bsz, past_len]
        
        # === Stage3: Select top tokens ===
        # Safety check: ensure budget_size doesn't exceed available tokens
        past_len = attn_cache_mean.shape[-1]
        actual_budget = min(self.budget_size, past_len)
        if actual_budget <= 0:
            # If no past tokens to compress, return full KV
            return None
        indices = attn_cache_mean.topk(actual_budget, dim=-1).indices
        indices = indices.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)
        
        # Gather compressed past KV
        k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
        
        # Concat with window
        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]
        
        key_states_out = torch.cat([k_past_compress, k_cur], dim=2)
        value_states_out = torch.cat([v_past_compress, v_cur], dim=2)
        
        return attn_cache, indices, key_states_out, value_states_out
    
    @staticmethod
    def count_elements(tensor, ga):
        cnts = torch.zeros(ga.shape[0], dtype=tensor.dtype).to(tensor.device)
        cnts = cnts.scatter_add_(0, tensor, torch.ones_like(tensor, dtype=tensor.dtype))
        return cnts
    
    def update_and_reset_budget(
        self,
        budget_k_cache: List[torch.Tensor],
        budget_v_cache: List[torch.Tensor],
        total_gather_indices: List[torch.Tensor],
        advance_attn_cache: List[torch.Tensor],
    ):
        """
        Cross-layer budget reallocation based on attention importance.
        Compatible with DynamicKV interface.
        """
        bz, head_num, kv_len, head_dim = budget_k_cache[0].shape
        budget_length_per_layer = [k.size(-2) - self.window_size for k in budget_k_cache]
        
        # Concatenate attention caches across layers
        gather_attn = torch.cat([t.unsqueeze(0) for t in advance_attn_cache], dim=0)
        flat_gather_attn = gather_attn.view(-1)
        
        tk = self.base_budget * head_num * gather_attn.shape[0]
        try:
            _, topk_indices = torch.topk(flat_gather_attn, k=tk)
        except:
            _, topk_indices = torch.topk(flat_gather_attn, k=flat_gather_attn.numel())
        
        if self.layer_idx == self.num_hidden_layers - 1:
            print(f"[CompilerKV] Top score: {_[0]:.4f}, Bottom score: {_[-1]:.4f}")
        
        # Count tokens per layer
        dim1 = gather_attn.shape[1] * gather_attn.shape[2] * gather_attn.shape[3]
        indices_0 = topk_indices // dim1
        counts = self.count_elements(indices_0, gather_attn) // gather_attn.shape[0]
        
        if torch.sum(counts).item() != dim1 and self.layer_idx == self.num_hidden_layers - 1:
            need_add = (tk // self.num_hidden_layers - torch.sum(counts))
            print(f"[CompilerKV] Need to add: {need_add}")
            counts[-1] += need_add
        
        # Compute per-layer budget
        norm_minv_per_layer = [t / torch.sum(counts).item() for t in counts]
        budget_length_fix = [int((self.budget_size * t).item()) for t in norm_minv_per_layer]
        
        need_fill_kv = self.base_budget * self.num_hidden_layers
        ss_radio = sum(budget_length_fix) / need_fill_kv if need_fill_kv > 0 else 1.0
        budget_length_fix = [int(k / ss_radio) if ss_radio > 0 else k for k in budget_length_fix]
        
        if sum(budget_length_fix) != need_fill_kv:
            budget_length_fix[-1] += need_fill_kv - sum(budget_length_fix)
        
        if self.layer_idx == self.num_hidden_layers - 1:
            print(f"[CompilerKV] Budget per layer: {budget_length_fix[:5]}...{budget_length_fix[-5:]}")
        
        # Check minimum budget constraint
        if min(budget_length_fix) < int(self.radio_min * self.base_budget):
            print(f"[CompilerKV] Warning: min budget {min(budget_length_fix)} < threshold")
        
        # Reallocate KV cache
        update_indices_list = []
        bgt_k_cache = []
        bgt_v_cache = []
        
        for k_cache, v_cache, fix_length in zip(budget_k_cache, budget_v_cache, budget_length_fix):
            if fix_length > gather_attn.shape[3]:
                fix_length = gather_attn.shape[3]
            
            k_cache_f = k_cache[:, :, :fix_length, :]
            v_cache_f = v_cache[:, :, :fix_length, :]
            k_cur = k_cache[:, :, -self.window_size:, :]
            v_cur = v_cache[:, :, -self.window_size:, :]
            
            key_states = torch.cat([k_cache_f, k_cur], dim=2)
            value_states = torch.cat([v_cache_f, v_cur], dim=2)
            
            bgt_k_cache.append(key_states)
            bgt_v_cache.append(value_states)
        
        return bgt_k_cache, bgt_v_cache, update_indices_list


def init_compilerkv(self, num_hidden_layers: int):
    """Initialize CompilerKV for attention module."""
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 64
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 7
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'radio_max'):
            self.config.radio_max = 10.0
        if not hasattr(self.config, 'tables_dir'):
            self.config.tables_dir = "Base/tables/outputs"
        
        self.kv_cluster = CompilerKVCluster(
            num_hidden_layers=num_hidden_layers,
            num_heads=getattr(self.config, 'num_attention_heads', 32),
            window_size=self.config.window_size,
            max_capacity_prompt=self.config.max_capacity_prompt,
            kernel_size=self.config.kernel_size,
            pooling=self.config.pooling,
            layer_idx=self.layer_idx,
            tables_dir=self.config.tables_dir,
            radio_max=self.config.radio_max,
        )
