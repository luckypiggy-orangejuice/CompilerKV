import torch
from typing import Optional, Tuple, Dict, Any
from importlib.metadata import version
import transformers

# DynamicKV V11 implementations
from .llama_model_impl.dynamickv_v11 import llama_flash_attn2_forward_DynamicKV_V11
from .mistral_model_impl.dynamickv_v11 import mistral_flash_attn2_forward_DynamicKV_V11
from .qwen2_model_impl.dynamickv_v11 import qwen2_flash_attn2_forward_DynamicKV_V11
from .internlm_model_impl.dynamickv_v11 import internlm_flash_attn2_forward_DynamicKV_V11

# CompilerKV implementations (Stage1 + Stage2 + Stage3)
from .llama_model_impl.compilerkv import llama_flash_attn2_forward_CompilerKV
from .mistral_model_impl.compilerkv import mistral_flash_attn2_forward_CompilerKV
from .qwen2_model_impl.compilerkv import qwen2_flash_attn2_forward_CompilerKV
from .internlm_model_impl.compilerkv import internlm_flash_attn2_forward_CompilerKV

from .llama_model_impl.utils import prepare_inputs_for_generation_llama
from .mistral_model_impl.utils import prepare_inputs_for_generation_mistral
from .qwen2_model_impl.utils import prepare_inputs_for_generation_qwen2
from .internlm_model_impl.utils import prepare_inputs_for_generation_internlm

import sys
sys.path.append('/root/.cache/huggingface/modules')
try:
    import transformers_modules.internlm2_5_7b_chat_1m.modeling_internlm2
except ImportError:
    pass  # InternLM module not available

llama_forward_function_map = {
    "dynamickv_v11": llama_flash_attn2_forward_DynamicKV_V11,
    "compilerkv": llama_flash_attn2_forward_CompilerKV,
}

mistral_forward_function_map = {
    "dynamickv_v11": mistral_flash_attn2_forward_DynamicKV_V11,
    "compilerkv": mistral_flash_attn2_forward_CompilerKV,
}

qwen2_forward_function_map = {
    "dynamickv_v11": qwen2_flash_attn2_forward_DynamicKV_V11,
    "compilerkv": qwen2_flash_attn2_forward_CompilerKV,
}

internlm_forward_function_map = {
    "dynamickv_v11": internlm_flash_attn2_forward_DynamicKV_V11,
    "compilerkv": internlm_flash_attn2_forward_CompilerKV,
}

def replace_attention(model_type: str, method: str):
    if method.lower() == "fullkv":
        return
    print(f"Using method: {method}!!!")
    print(f"Replacing attention forward: {model_type}!!!")
    
    method_lower = method.lower()
    
    if model_type == "llama" or model_type == "lwm":
        if method_lower in llama_forward_function_map:
            transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_forward_function_map[method_lower]
        else:
            raise ValueError(f"Unknown method for llama: {method}")
    elif model_type == "mistral":
        if method_lower in mistral_forward_function_map:
            transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_forward_function_map[method_lower]
        else:
            raise ValueError(f"Unknown method for mistral: {method}")
    elif model_type == "qwen2":
        if method_lower in qwen2_forward_function_map:
            transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_forward_function_map[method_lower]
        else:
            raise ValueError(f"Unknown method for qwen2: {method}")
    elif model_type == "internlm":
        if method_lower in internlm_forward_function_map:
            try:
                transformers_modules.internlm2_5_7b_chat_1m.modeling_internlm2.InternLM2FlashAttention2.forward = internlm_forward_function_map[method_lower]
            except NameError:
                raise ValueError("InternLM module not available. Please ensure it's properly installed.")
        else:
            raise ValueError(f"Unknown method for internlm: {method}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    if method_lower not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral
        transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen2
        try:
            transformers_modules.internlm2_5_7b_chat_1m.modeling_internlm2.InternLM2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_internlm
        except NameError:
            pass  # InternLM not available

    
def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version