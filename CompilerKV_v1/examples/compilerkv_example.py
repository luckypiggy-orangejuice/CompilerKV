"""
CompilerKV Usage Example

This example shows how to use the CompilerKV method for KV cache compression
during inference with LLaMA models.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kv_compression.token_drop.monkeypatch import replace_attention


def main():
    # Configuration
    model_path = "models/Llama-3-8B-Instruct"  # Path to model
    method = "compilerkv"  # Our Stage1-Stage2-Stage3 method
    
    # KV compression parameters
    max_capacity_prompt = 2048  # Total KV budget per layer
    window_size = 64            # Observation window size
    kernel_size = 7             # Smoothing kernel
    pooling = "avgpool"         # Pooling method
    radio_max = 10.0            # Max budget ratio
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    model.eval()
    
    # Replace attention with CompilerKV
    print(f"Applying {method} KV compression...")
    replace_attention(model_type="llama", method=method)
    
    # Configure compression parameters for each layer
    num_layers = model.config.num_hidden_layers
    for i in range(num_layers):
        layer = model.model.layers[i].self_attn
        layer.config.window_size = window_size
        layer.config.max_capacity_prompt = max_capacity_prompt
        layer.config.kernel_size = kernel_size
        layer.config.pooling = pooling
        layer.config.radio_max = radio_max
        layer.config.tables_dir = "Base/tables/outputs"
    
    # Test inference
    print("\n" + "="*50)
    print("Testing inference with CompilerKV compression")
    print("="*50)
    
    # Example prompt (you can use a longer context for better demonstration)
    prompt = """You are a helpful assistant. Please answer the following question concisely.

Context: The Python programming language was conceived in the late 1980s by Guido van Rossum 
at Centrum Wiskunde & Informatica (CWI) in the Netherlands as a successor to the ABC programming 
language, which was inspired by SETL, capable of exception handling and interfacing with the 
Amoeba operating system. Its implementation began in December 1989. Van Rossum shouldered sole 
responsibility for the project, as the lead developer, until 12 July 2018, when he announced 
his "permanent vacation" from his responsibilities as Python's chief architect.

Question: Who created Python and when did they announce their permanent vacation?

Answer:"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print(f"Input length: {inputs['input_ids'].shape[1]} tokens")
    print(f"KV budget per layer: {max_capacity_prompt} tokens")
    print(f"Window size: {window_size} tokens")
    
    # Generate
    print("\nGenerating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )
    
    # Decode and print response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print("\n" + "="*50)
    print("Generated Response:")
    print("="*50)
    print(response)
    print("="*50)


if __name__ == "__main__":
    main()
