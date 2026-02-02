"""
CompilerKV: Prefill-only KV Cache Compression

This package implements the Stage1-Stage2-Stage3 KV compression method:
- Stage1: Token utility computation (α_t * ρ_t)
- Stage2: Head-aware reweighting using W_head table
- Stage3: Risk-adaptive threshold gating with M_lex LUT + budget fix

Usage:
    # Option 1: Use monkeypatch to replace attention
    from kv_compression.token_drop.monkeypatch import replace_attention
    replace_attention(model_type="llama", method="compilerkv")
    
    # Option 2: Use API directly
    from kv_compression.api import compress_kv_prefill_only
    K_comp, V_comp, indices = compress_kv_prefill_only(attn, V, budgets)
"""

from .api import compress_kv_prefill_only, CompilerKVCompressor, KVCompressor

__all__ = [
    "compress_kv_prefill_only",
    "CompilerKVCompressor",
    "KVCompressor",
]
