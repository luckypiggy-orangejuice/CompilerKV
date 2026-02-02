
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import jsonlines
import torch

class DynamicKVClusterV11():
    def __init__(self, 
                 num_hidden_layers = 32, 
                 window_size = 64, 
                 max_capacity_prompt = 256 + 64, 
                 kernel_size = 7, 
                 pooling = 'avgpool', 
                 layer_idx=None,
                 radio_max = 10,
                 radio_min = 0.1
                 ):
        
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.base = self.max_capacity_prompt - self.window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.radio_max = radio_max
        self.radio_min = radio_min
        self.budget_size = -1

    
    def budget_compute_per_layer(self, 
                  key_states, 
                  query_states, 
                  value_states, 
                  attention_mask):
        
        bsz, num_heads, q_len, head_dim = query_states.shape
        if self.layer_idx == 0: 
            print(f"DynamicKV max capacity prompt: {self.max_capacity_prompt} \
                    window size: {self.window_size} \
                    base: {self.base} radio_max: {self.radio_max}")
        if q_len < self.window_size:
            return None
        else:
            if self.budget_size==-1: self.budget_size = min(int(self.radio_max * self.base), (q_len - self.window_size))
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]
            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool': attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool': attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else: raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.budget_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return attn_cache, indices, key_states, value_states

    @staticmethod
    def count_elements(tensor, ga):
        cnts = torch.zeros(ga.shape[0], dtype=tensor.dtype).to(tensor.device)
        cnts = cnts.scatter_add_(0, tensor, torch.ones_like(tensor, dtype=tensor.dtype))
        return cnts

    def update_and_reset_budget(self, 
                                budget_k_cache, 
                                budget_v_cache, 
                                total_gather_indices,
                                advance_attn_cache):
        bz, head_num, kv_len, head_dim = budget_k_cache[0].shape
        budget_length_per_layer = [k.size(-2)-self.window_size for k in budget_k_cache]

        gather_attn = torch.cat(([t.unsqueeze(0) for t in advance_attn_cache]), dim=0) 
        flat_gather_attn = gather_attn.view(-1)
        tk = self.base * head_num* gather_attn.shape[0]
        try:
            _, topk_indices = torch.topk(flat_gather_attn, k=tk)
        except:
            _, topk_indices = torch.topk(flat_gather_attn, k=flat_gather_attn.shape.numel())
        if self.layer_idx == self.num_hidden_layers-1:
            print(f"first number of _: {_[0]}")
            print(f"last number of _: {_[-1]}")
        dim1 = gather_attn.shape[1] * gather_attn.shape[2] * gather_attn.shape[3]  # 1*32*8000
        indices_0 = topk_indices // dim1
        counts = self.count_elements(indices_0, gather_attn) // gather_attn.shape[0]
        if torch.sum(counts).item() != dim1 and self.layer_idx == self.num_hidden_layers-1:
            need_add = (tk//self.num_hidden_layers - torch.sum(counts))
            print("需要增加的数量：",need_add)
            counts[-1] += need_add

        norm_minv_per_layer = [t/torch.sum(counts).item() for t in counts]

        budget_length_fix = [int((self.budget_size * t).item()) for t in norm_minv_per_layer]
        need_fill_kv = self.base* self.num_hidden_layers
        ss_radio = sum([int((self.budget_size * t).item()) for t in norm_minv_per_layer]) / need_fill_kv
        budget_length_fix = [int(k // ss_radio) for k in budget_length_fix]
        if sum(budget_length_fix) != need_fill_kv:
            try:
                budget_length_fix[-1] += need_fill_kv - sum(budget_length_fix)
            except:
                budget_length_fix[-1] = budget_length_fix[-1]
        if self.layer_idx != self.num_hidden_layers:
            print("mid_budget_length_fix: ", budget_length_fix)
        else: 
            print("budget_length_fix: ", budget_length_fix)
        update_indices_list = []
        bgt_k_cache = []
        bgt_v_cache = []
        
        if min(budget_length_fix) < int(self.radio_min * self.base):
            print(f"Attention!!! The min number of budget_length_fix is {min(budget_length_fix)}")
        for k_cache, v_cache, fix_length in zip(budget_k_cache, budget_v_cache, budget_length_fix):
            if fix_length > gather_attn.shape[3]:
                fix_length = gather_attn.shape[3]
            k_cache_f = k_cache[:,:,:fix_length,:]
            v_cache_f = v_cache[:,:,:fix_length,:]
            k_cur = k_cache[:, :, -self.window_size:, :]
            v_cur = v_cache[:, :, -self.window_size:, :]
            key_states = torch.cat([k_cache_f, k_cur], dim = 2)
            value_states = torch.cat([v_cache_f, v_cur], dim = 2)
            bgt_k_cache.append(key_states)
            bgt_v_cache.append(value_states)
        
        return bgt_k_cache, bgt_v_cache, update_indices_list



def init_dynamickv_V11(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 8
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'radio_max'):
            self.config.radio_max = 10
    
    
    self.kv_cluster = DynamicKVClusterV11(
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        radio_max=self.config.radio_max
        )
