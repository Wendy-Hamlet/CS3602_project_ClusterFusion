#!/usr/bin/env python3
"""
End-to-End Benchmark for ClusterFusion Attention Kernel (Pythia-2.8B)

Measures Time Per Output Token (TPOT) comparing:
- CUDA Attention + PyTorch MLP Down (ClusterFusion)
- Full PyTorch baseline
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "EleutherAI/pythia-2.8b"
TOKEN_COUNTS = [16, 32, 64, 128, 256, 512, 1024, 2048]
PROMPT = "The meaning of life is"

# Model configuration
HIDDEN_DIM = 2560
NUM_HEADS = 32
HEAD_DIM = 80
ROTARY_DIM = 20
FFN_DIM = 10240
NUM_LAYERS = 32


def precompute_rope(max_pos, device="cuda:0"):
    """Precompute rotary position embeddings."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, ROTARY_DIM, 2, device=device).float() / ROTARY_DIM))
    positions = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    cos = torch.cat([cos, torch.ones((max_pos, HEAD_DIM - ROTARY_DIM), device=device)], -1)
    sin = torch.cat([sin, torch.zeros((max_pos, HEAD_DIM - ROTARY_DIM), device=device)], -1)
    return cos, sin


def pytorch_attention_mlp_up(hidden, layer, k_cache, v_cache, cos, sin, seq_len):
    """PyTorch baseline for Attention + MLP Up + GELU."""
    # LayerNorm
    ln_out = F.layer_norm(hidden, (HIDDEN_DIM,), 
                          layer.input_layernorm.weight, 
                          layer.input_layernorm.bias)
    
    # QKV Projection
    qkv = F.linear(ln_out, layer.attention.query_key_value.weight, 
                   layer.attention.query_key_value.bias)
    q, k, v = qkv.chunk(3, dim=-1)
    
    # Reshape for multi-head attention
    q = q.view(1, 1, NUM_HEADS, HEAD_DIM)
    k = k.view(1, 1, NUM_HEADS, HEAD_DIM)
    v = v.view(1, 1, NUM_HEADS, HEAD_DIM)
    
    # RoPE
    cos_pos = cos[seq_len].view(1, 1, 1, HEAD_DIM)
    sin_pos = sin[seq_len].view(1, 1, 1, HEAD_DIM)
    
    def apply_rope(x):
        x_rot = x[..., :ROTARY_DIM]
        x_pass = x[..., ROTARY_DIM:]
        x1 = x_rot[..., :ROTARY_DIM//2]
        x2 = x_rot[..., ROTARY_DIM//2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        x_rotated = x_rot * cos_pos[..., :ROTARY_DIM] + rotated * sin_pos[..., :ROTARY_DIM]
        return torch.cat([x_rotated, x_pass], dim=-1)
    
    q = apply_rope(q)
    k = apply_rope(k)
    
    # Update KV cache
    k_flat = k.view(1, HIDDEN_DIM)
    v_flat = v.view(1, HIDDEN_DIM)
    k_cache[seq_len] = k_flat
    v_cache[seq_len] = v_flat
    
    # Attention
    k_cached = k_cache[:seq_len + 1].view(seq_len + 1, NUM_HEADS, HEAD_DIM)
    v_cached = v_cache[:seq_len + 1].view(seq_len + 1, NUM_HEADS, HEAD_DIM)
    
    q = q.squeeze(0).squeeze(0)  # [NUM_HEADS, HEAD_DIM]
    
    attn_scores = torch.einsum('hd,shd->hs', q.float(), k_cached.float()) / (HEAD_DIM ** 0.5)
    attn_probs = F.softmax(attn_scores, dim=-1).half()
    attn_out = torch.einsum('hs,shd->hd', attn_probs.float(), v_cached.float()).half()
    attn_out = attn_out.view(1, HIDDEN_DIM)
    
    # Output projection
    attn_output = F.linear(attn_out, layer.attention.dense.weight, 
                           layer.attention.dense.bias)
    
    # Post-attention LayerNorm (Pythia uses parallel residual)
    post_ln_out = F.layer_norm(hidden.squeeze(0), (HIDDEN_DIM,),
                               layer.post_attention_layernorm.weight,
                               layer.post_attention_layernorm.bias)
    
    # MLP Up + GELU
    mlp_intermediate = F.linear(post_ln_out, layer.mlp.dense_h_to_4h.weight,
                                layer.mlp.dense_h_to_4h.bias)
    mlp_intermediate = F.gelu(mlp_intermediate)
    
    return attn_output, mlp_intermediate


def prepare_state(model, tokenizer, prompt, num_new_tokens):
    """Prepare state for benchmarking (prefill + weight extraction)."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]
    
    # Prefill
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        first_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
    max_seq_len = prompt_length + num_new_tokens
    
    # Extract weights and prepare caches
    all_weights = []
    kv_caches = []
    for layer_idx in range(NUM_LAYERS):
        layer = model.gpt_neox.layers[layer_idx]
        weights = {
            "ln_weight": layer.input_layernorm.weight.contiguous(),
            "ln_bias": layer.input_layernorm.bias.contiguous(),
            "qkv_weight": layer.attention.query_key_value.weight.T.contiguous(),
            "qkv_bias": layer.attention.query_key_value.bias.contiguous(),
            "o_weight": layer.attention.dense.weight.T.contiguous(),
            "o_bias": layer.attention.dense.bias.contiguous(),
            "post_ln_weight": layer.post_attention_layernorm.weight.contiguous(),
            "post_ln_bias": layer.post_attention_layernorm.bias.contiguous(),
            "mlp_up_weight": layer.mlp.dense_h_to_4h.weight.T.contiguous(),
            "mlp_up_bias": layer.mlp.dense_h_to_4h.bias.contiguous(),
            "mlp_down_weight": layer.mlp.dense_4h_to_h.weight.half(),
            "mlp_down_bias": layer.mlp.dense_4h_to_h.bias.half(),
        }
        all_weights.append(weights)
        
        k = past_key_values[layer_idx][0].squeeze(0).transpose(0, 1).contiguous()
        v = past_key_values[layer_idx][1].squeeze(0).transpose(0, 1).contiguous()
        k = k.reshape(k.shape[0], -1)
        v = v.reshape(v.shape[0], -1)
        
        k_cache = torch.zeros((max_seq_len, HIDDEN_DIM), dtype=torch.float16, device=device)
        v_cache = torch.zeros((max_seq_len, HIDDEN_DIM), dtype=torch.float16, device=device)
        k_cache[:k.shape[0]] = k
        v_cache[:v.shape[0]] = v
        kv_caches.append((k_cache, v_cache, k.shape[0]))
    
    return {
        "input_ids": input_ids,
        "prompt_length": prompt_length,
        "first_token": first_token,
        "all_weights": all_weights,
        "kv_caches": kv_caches,
    }


def decode_clusterfusion(model, num_new_tokens, state):
    """Decode using CUDA Attention + PyTorch MLP Down."""
    import clusterfusion
    
    device = next(model.parameters()).device
    next_token = state["first_token"]
    prompt_length = state["prompt_length"]
    all_weights = state["all_weights"]
    kv_caches = [(k.clone(), v.clone(), l) for k, v, l in state["kv_caches"]]
    
    max_position = prompt_length + num_new_tokens
    all_cos, all_sin = precompute_rope(max_position, device)
    
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for step in range(num_new_tokens - 1):
            current_pos = prompt_length + step
            hidden = model.gpt_neox.embed_in(next_token).half().squeeze(1)
            cos = all_cos[current_pos]
            sin = all_sin[current_pos]
            
            for layer_idx in range(NUM_LAYERS):
                w = all_weights[layer_idx]
                k_cache, v_cache, cur_len = kv_caches[layer_idx]
                input_residual = hidden.clone()
                
                # CUDA kernel: Attention + MLP Up
                attn_out, mlp_int, _, _ = clusterfusion.pythia_2b8_attention_only(
                    hidden, w["qkv_weight"], w["qkv_bias"], w["o_weight"], w["o_bias"],
                    k_cache, v_cache, w["ln_weight"], w["ln_bias"], cos, sin,
                    w["post_ln_weight"], w["post_ln_bias"], w["mlp_up_weight"], w["mlp_up_bias"],
                    cur_len
                )
                
                # PyTorch: MLP Down + Residual
                mlp_down = F.linear(mlp_int.unsqueeze(0), w["mlp_down_weight"], w["mlp_down_bias"])
                hidden = input_residual + attn_out + mlp_down
                kv_caches[layer_idx] = (k_cache, v_cache, cur_len + 1)
            
            hidden = F.layer_norm(hidden, (HIDDEN_DIM,),
                                  model.gpt_neox.final_layer_norm.weight,
                                  model.gpt_neox.final_layer_norm.bias)
            logits = model.embed_out(hidden)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
    
    torch.cuda.synchronize()
    return time.time() - start


def decode_pytorch(model, num_new_tokens, state):
    """Decode using full PyTorch baseline."""
    device = next(model.parameters()).device
    next_token = state["first_token"]
    prompt_length = state["prompt_length"]
    all_weights = state["all_weights"]
    kv_caches = [(k.clone(), v.clone(), l) for k, v, l in state["kv_caches"]]
    
    max_position = prompt_length + num_new_tokens
    all_cos, all_sin = precompute_rope(max_position, device)
    
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for step in range(num_new_tokens - 1):
            current_pos = prompt_length + step
            hidden = model.gpt_neox.embed_in(next_token).half().squeeze(1)
            
            for layer_idx in range(NUM_LAYERS):
                layer = model.gpt_neox.layers[layer_idx]
                w = all_weights[layer_idx]
                k_cache, v_cache, cur_len = kv_caches[layer_idx]
                input_residual = hidden.clone()
                
                # Full PyTorch: Attention + MLP Up
                attn_out, mlp_int = pytorch_attention_mlp_up(
                    hidden, layer, k_cache, v_cache, all_cos, all_sin, cur_len
                )
                
                # PyTorch: MLP Down + Residual
                mlp_down = F.linear(mlp_int, w["mlp_down_weight"], w["mlp_down_bias"])
                hidden = input_residual + attn_out + mlp_down
                kv_caches[layer_idx] = (k_cache, v_cache, cur_len + 1)
            
            hidden = F.layer_norm(hidden, (HIDDEN_DIM,),
                                  model.gpt_neox.final_layer_norm.weight,
                                  model.gpt_neox.final_layer_norm.bias)
            logits = model.embed_out(hidden)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
    
    torch.cuda.synchronize()
    return time.time() - start


def main():
    print("=" * 90)
    print("Pythia-2.8B End-to-End Benchmark: ClusterFusion vs PyTorch")
    print("=" * 90)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Model: {MODEL_NAME}")
    print(f"Params: hidden={HIDDEN_DIM}, heads={NUM_HEADS}, head_dim={HEAD_DIM}, layers={NUM_LAYERS}")
    
    # Warmup
    print("\nWarming up...")
    warmup_state = prepare_state(model, tokenizer, PROMPT, 8)
    decode_clusterfusion(model, 8, warmup_state)
    warmup_state = prepare_state(model, tokenizer, PROMPT, 8)
    decode_pytorch(model, 8, warmup_state)
    torch.cuda.synchronize()
    
    # Benchmark
    results = []
    for num_tokens in TOKEN_COUNTS:
        state = prepare_state(model, tokenizer, PROMPT, num_tokens)
        cf_time = decode_clusterfusion(model, num_tokens, state)
        
        state = prepare_state(model, tokenizer, PROMPT, num_tokens)
        pt_time = decode_pytorch(model, num_tokens, state)
        
        speedup = pt_time / cf_time if cf_time > 0 else float("inf")
        tpot_cf = cf_time / (num_tokens - 1) * 1000  # ms per token
        tpot_pt = pt_time / (num_tokens - 1) * 1000  # ms per token
        
        results.append({
            "tokens": num_tokens,
            "cf_time": cf_time,
            "pt_time": pt_time,
            "speedup": speedup,
            "tpot_cf": tpot_cf,
            "tpot_pt": tpot_pt,
        })
    
    # Results
    print("\n" + "=" * 90)
    print("Results (decode time only, excluding prefill)")
    print("=" * 90)
    print(f"{'Tokens':>8} | {'CF(s)':>8} | {'PyTorch(s)':>10} | {'Speedup':>8} | {'TPOT CF(ms)':>12} | {'TPOT PT(ms)':>12}")
    print("-" * 90)
    for r in results:
        print(f"{r['tokens']:>8} | {r['cf_time']:>8.3f} | {r['pt_time']:>10.3f} | {r['speedup']:>7.2f}x | {r['tpot_cf']:>12.2f} | {r['tpot_pt']:>12.2f}")
    
    # Summary
    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    avg_tpot_cf = sum(r['tpot_cf'] for r in results) / len(results)
    avg_tpot_pt = sum(r['tpot_pt'] for r in results) / len(results)
    print(f"Average Speedup:     {avg_speedup:.2f}x")
    print(f"Max Speedup:         {max(r['speedup'] for r in results):.2f}x")
    print(f"Average TPOT (CF):   {avg_tpot_cf:.2f} ms")
    print(f"Average TPOT (PT):   {avg_tpot_pt:.2f} ms")


if __name__ == "__main__":
    main()
