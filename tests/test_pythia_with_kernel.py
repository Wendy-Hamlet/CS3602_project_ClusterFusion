"""
Full Pythia decoder layer using ClusterFusion kernel.
Kernel handles: LayerNorm -> Attention -> Output Proj -> Post-LN -> MLP -> Parallel Residual
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import clusterfusion
import time

MODEL_NAME = "EleutherAI/pythia-2.8b"

def compute_rope_embeddings(position, rotary_dim, head_dim, base=10000, device='cuda:0'):
    """
    Compute RoPE embeddings matching HuggingFace.
    Kernel expects HEAD_DIM size, so we pad with identity (cos=1, sin=0).
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    position_tensor = torch.tensor([position], dtype=torch.float32, device=device)
    freqs = torch.outer(position_tensor, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    
    # Pad to HEAD_DIM size with identity (cos=1, sin=0)
    padding_size = head_dim - rotary_dim
    cos_padding = torch.ones((1, padding_size), dtype=torch.float32, device=device)
    sin_padding = torch.zeros((1, padding_size), dtype=torch.float32, device=device)
    
    cos = torch.cat([cos, cos_padding], dim=-1)
    sin = torch.cat([sin, sin_padding], dim=-1)
    
    return cos, sin

def generate_with_kernel(
    model,
    tokenizer,
    prompt,
    num_new_tokens,
    all_weights,
    kv_caches,
    input_ids,
    prompt_length,
    first_token,
):
    """Generate using ClusterFusion kernel for the full decoder (attention + MLP)."""
    print(f"\\n{'='*80}")
    print(f"ClusterFusion Kernel (Full Decoder Layer with MLP)")
    print(f"Prompt: '{prompt}', Generating {num_new_tokens} tokens")
    print(f"{'='*80}\\n")
    
    device = next(model.parameters()).device

    next_token = first_token
    generated_ids = [next_token.item()]
    print(f"First token: {next_token.item()} ('{tokenizer.decode([next_token.item()])}')")
    
    # Model constants
    num_layers = len(model.gpt_neox.layers)
    num_heads = 32
    head_dim = 80
    hidden_size = 2560
    rotary_dim = 20
    
    # Decoding
    for step in range(num_new_tokens - 1):
        current_position = prompt_length + step
        
        # Embedding
        hidden_states = model.gpt_neox.embed_in(next_token).half().squeeze(1)
        
        # RoPE embeddings
        cos, sin = compute_rope_embeddings(current_position, rotary_dim, head_dim, base=10000, device=device)
        
        # Through all layers
        for layer_idx in range(num_layers):
            weights = all_weights[layer_idx]
            k_cache_full, v_cache_full, current_len = kv_caches[layer_idx]
            
            # ========== Full Decoder Layer with ClusterFusion Kernel ==========
            # Kernel handles: LayerNorm -> Attention -> Output Proj -> Post-LN -> MLP -> Residual
            hidden_states, new_k, new_v = clusterfusion.pythia_decoder_layer(
                hidden_states,
                weights['qkv_weight'],
                weights['qkv_bias'],
                weights['o_weight'],
                weights['o_bias'],
                k_cache_full,
                v_cache_full,
                weights['ln_weight'],
                weights['ln_bias'],
                cos,
                sin,
                # MLP weights
                weights['post_ln_weight'],
                weights['post_ln_bias'],
                weights['mlp_up_weight'],
                weights['mlp_up_bias'],
                weights['mlp_down_weight'],
                weights['mlp_down_bias'],
                current_len  # Current sequence length
            )
            
            # Update current length (kernel already wrote to cache[current_len])
            kv_caches[layer_idx] = (k_cache_full, v_cache_full, current_len + 1)
        
        # Final LayerNorm
        hidden_states = F.layer_norm(
            hidden_states, (hidden_size,),
            model.gpt_neox.final_layer_norm.weight.data,
            model.gpt_neox.final_layer_norm.bias.data,
            eps=1e-5
        )
        
        # Logits
        logits = model.embed_out(hidden_states)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids.append(next_token.item())
    
    full_ids = input_ids[0].tolist() + generated_ids
    generated_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    
    print(f"{'='*80}")
    print(f"Generated text:\\n{generated_text}")
    print(f"{'='*80}")
    
    return generated_text, generated_ids

if __name__ == "__main__":
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map='cuda:0')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    prompt = "The meaning of life is"
    num_tokens = 20

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]

    # ==================== Setup (excluded from timing) ====================
    with torch.no_grad():
        torch.cuda.synchronize()
        start_setup = time.time()

        # Prefill with HF to get initial KV cache and first token
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        first_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Extract weights once
        num_layers = len(model.gpt_neox.layers)
        hidden_size = 2560
        max_seq_len = prompt_length + num_tokens  # prefill + decode tokens
        all_weights = []
        kv_caches = []
        for layer_idx in range(num_layers):
            layer = model.gpt_neox.layers[layer_idx]
            weights = {
                'ln_weight': layer.input_layernorm.weight.data.unsqueeze(0).half(),
                'ln_bias': layer.input_layernorm.bias.data.unsqueeze(0).half(),
                'qkv_weight': layer.attention.query_key_value.weight.data.half(),
                'qkv_bias': layer.attention.query_key_value.bias.data.half(),
                'o_weight': layer.attention.dense.weight.data.half(),
                'o_bias': layer.attention.dense.bias.data.half(),
                'post_ln_weight': layer.post_attention_layernorm.weight.data.unsqueeze(0).half(),
                'post_ln_bias': layer.post_attention_layernorm.bias.data.unsqueeze(0).half(),
                'mlp_up_weight': layer.mlp.dense_h_to_4h.weight.data.half(),
                'mlp_up_bias': layer.mlp.dense_h_to_4h.bias.data.half(),
                'mlp_down_weight': layer.mlp.dense_4h_to_h.weight.data.half(),
                'mlp_down_bias': layer.mlp.dense_4h_to_h.bias.data.half(),
            }
            all_weights.append(weights)

            k = past_key_values[layer_idx][0].squeeze(0).transpose(0, 1).contiguous()
            v = past_key_values[layer_idx][1].squeeze(0).transpose(0, 1).contiguous()
            k = k.reshape(k.shape[0], -1)
            v = v.reshape(v.shape[0], -1)

            k_cache_full = torch.zeros((max_seq_len, hidden_size), dtype=torch.float16, device=device)
            v_cache_full = torch.zeros((max_seq_len, hidden_size), dtype=torch.float16, device=device)
            k_cache_full[:k.shape[0]] = k
            v_cache_full[:v.shape[0]] = v

            kv_caches.append((k_cache_full, v_cache_full, k.shape[0]))

        torch.cuda.synchronize()
        setup_time_kernel = time.time() - start_setup

    # ==================== ClusterFusion decode timing ====================
    torch.cuda.synchronize()
    start = time.time()
    text_kernel, ids_kernel = generate_with_kernel(
        model,
        tokenizer,
        prompt,
        num_tokens,
        all_weights,
        kv_caches,
        input_ids,
        prompt_length,
        first_token,
    )
    torch.cuda.synchronize()
    time_kernel = time.time() - start
    
    # HuggingFace reference (decode timing only)
    print(f"{'='*80}")
    print(f"HuggingFace Reference")
    print(f"{'='*80}")
    
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        output_ids_hf = model.generate(input_ids, max_new_tokens=num_tokens, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        time_hf = time.time() - start
    
    text_hf = tokenizer.decode(output_ids_hf[0], skip_special_tokens=True)
    ids_hf = output_ids_hf[0].tolist()
    
    print(f"Generated text:{text_hf}")
    
    # Compare
    print(f"{'='*80}")
    print(f"Comparison")
    print(f"{'='*80}")
    print(f"ClusterFusion setup time (excluded from decode): {setup_time_kernel:.3f}s")
    print(f"ClusterFusion (full decoder) decode time: {time_kernel:.3f}s")
    print(f"HuggingFace decode time: {time_hf:.3f}s")
    print(f"Speedup (decode only): {time_hf/time_kernel:.2f}x")
    print(f"Text match: {text_kernel == text_hf}")
    
    ids_kernel_full = input_ids[0].tolist() + ids_kernel
    print(f"Token IDs match: {ids_kernel_full == ids_hf}")
    
    if ids_kernel_full != ids_hf:
        print(f"Kernel IDs: {ids_kernel_full}")
        print(f"HuggingFace IDs: {ids_hf}")
        
        for i, (k, h) in enumerate(zip(ids_kernel_full, ids_hf)):
            if k != h:
                print(f"  First mismatch at position {i}:")
                print(f"  Kernel: {k} ('{tokenizer.decode([k])}')")
                print(f"  HuggingFace: {h} ('{tokenizer.decode([h])}')")
                break

