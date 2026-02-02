import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
)
import inspect
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import (
    logging,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from ..methods.dynamic_v11 import init_dynamickv_V11
logger = logging.get_logger(__name__)

budget_indices = []
budget_k_cache = []
budget_v_cache = []
budget_kv_cur_arrange = []
advance_attn_cache = []
total_gather_indices = []

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)



def qwen2_flash_attn2_forward_DynamicKV_V11(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    init_dynamickv_V11(self, num_hidden_layers=self.config.num_hidden_layers)
    global budget_k_cache
    global budget_v_cache
    global budget_indices
    global budget_kv_cur_arrange
    global advance_attn_cache
    global total_gather_indices


    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
            
    rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
    cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    
    use_sliding_windows = (
        _flash_supports_window_size
        and getattr(self.config, "sliding_window", None) is not None
        and kv_seq_len > self.config.sliding_window
        and self.config.use_sliding_window
    )

    if not _flash_supports_window_size:
        logger.warning_once(
            "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
            " make sure to upgrade flash-attn library."
        )
        
        
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = self.attention_dropout if self.training else 0.0
    
    if past_key_value is not None:
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (
            getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()


            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)
        
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        if key_states.shape[-2] == kv_seq_len: 
            self.kv_seq_len = kv_seq_len 
            if self.layer_idx == 0:
                budget_k_cache = []
                budget_v_cache = []
                budget_indices = []
                advance_attn_cache = []
                total_gather_indices = []
                
            cur_attn_cache, cur_indices_, cur_budget_k_cache, cur_budget_v_cache = self.kv_cluster.budget_compute_per_layer(
                key_states, 
                query_states, 
                value_states, 
                attention_mask)
            budget_k_cache.append(cur_budget_k_cache)
            budget_v_cache.append(cur_budget_v_cache)
            budget_indices.append(cur_indices_)
            advance_attn_cache.append(cur_attn_cache)
            if (self.layer_idx % 4) == 3:
                reallocate_budget_k_cache, reallocate_budget_v_cache, reallocate_gather_indices = self.kv_cluster.update_and_reset_budget(
                    budget_k_cache, 
                    budget_v_cache, 
                    total_gather_indices,
                    advance_attn_cache
                    )
                budget_k_cache = reallocate_budget_k_cache
                budget_v_cache = reallocate_budget_v_cache
                total_gather_indices = reallocate_gather_indices
                
            if self.layer_idx == self.config.num_hidden_layers-1:
                for t in range(self.config.num_hidden_layers):
                    past_key_value.update(budget_k_cache[t], budget_v_cache[t], t, cache_kwargs)

        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    
    
    

    dropout_rate = self.attention_dropout if self.training else 0.0
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)
    attn_output = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate, use_sliding_windows=use_sliding_windows
    )
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
