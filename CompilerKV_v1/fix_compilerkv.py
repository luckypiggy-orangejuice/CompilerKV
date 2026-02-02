import re

# Read the file
with open('kv_compression/token_drop/methods/compilerkv.py', 'r') as f:
    content = f.read()

# Fix 1: Reset budget_size at the start of each sample
old_code = '''        if q_len < self.window_size:
            return None
        
        # Initialize budget size
        if self.budget_size == -1:
            self.budget_size = min(
                int(self.radio_max * self.base_budget), 
                (q_len - self.window_size)
            )'''

new_code = '''        if q_len < self.window_size:
            return None
        
        # Compute budget size for this sample (reset for each new sample)
        past_len = q_len - self.window_size
        self.budget_size = min(
            int(self.radio_max * self.base_budget), 
            past_len
        )'''

content = content.replace(old_code, new_code)

# Fix 2: Add safety check for topk
old_code2 = '''        # === Stage3: Select top tokens ===
        indices = attn_cache_mean.topk(self.budget_size, dim=-1).indices'''

new_code2 = '''        # === Stage3: Select top tokens ===
        # Safety check: ensure budget_size doesn't exceed available tokens
        past_len = attn_cache_mean.shape[-1]
        actual_budget = min(self.budget_size, past_len)
        if actual_budget <= 0:
            # If no past tokens to compress, return full KV
            return None
        indices = attn_cache_mean.topk(actual_budget, dim=-1).indices'''

content = content.replace(old_code2, new_code2)

# Write the fixed content
with open('kv_compression/token_drop/methods/compilerkv.py', 'w') as f:
    f.write(content)

print("Fixed compilerkv.py successfully!")
