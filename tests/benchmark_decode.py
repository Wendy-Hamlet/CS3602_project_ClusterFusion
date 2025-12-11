"""
Benchmark ClusterFusion full decoder (attention + MLP) vs HuggingFace across
different numbers of generated tokens. Timing excludes one-time setup
(weight extraction, cache allocation, prefill). Only decode time is compared.
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import clusterfusion

MODEL_NAME = "EleutherAI/pythia-2.8b"
TOKEN_COUNTS = [16, 32, 64, 128, 256, 512, 1024, 2048]
PROMPT = "The meaning of life is"


def compute_rope_embeddings(position, rotary_dim, head_dim, base=10000, device="cuda:0"):
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    position_tensor = torch.tensor([position], dtype=torch.float32, device=device)
    freqs = torch.outer(position_tensor, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    # pad to HEAD_DIM
    padding_size = head_dim - rotary_dim
    cos = torch.cat([cos, torch.ones((1, padding_size), device=device)], dim=-1)
    sin = torch.cat([sin, torch.zeros((1, padding_size), device=device)], dim=-1)
    return cos, sin


def prepare_setup(model, tokenizer, prompt, num_new_tokens):
    """Prefill + weight extraction + cache allocation. Returns setup_time and state."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        first_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    num_layers = len(model.gpt_neox.layers)
    hidden_size = 2560
    max_seq_len = prompt_length + num_new_tokens

    all_weights = []
    kv_caches = []
    for layer_idx in range(num_layers):
        layer = model.gpt_neox.layers[layer_idx]
        weights = {
            "ln_weight": layer.input_layernorm.weight.data.unsqueeze(0).half(),
            "ln_bias": layer.input_layernorm.bias.data.unsqueeze(0).half(),
            "qkv_weight": layer.attention.query_key_value.weight.data.half(),
            "qkv_bias": layer.attention.query_key_value.bias.data.half(),
            "o_weight": layer.attention.dense.weight.data.half(),
            "o_bias": layer.attention.dense.bias.data.half(),
            "post_ln_weight": layer.post_attention_layernorm.weight.data.unsqueeze(0).half(),
            "post_ln_bias": layer.post_attention_layernorm.bias.data.unsqueeze(0).half(),
            "mlp_up_weight": layer.mlp.dense_h_to_4h.weight.data.half(),
            "mlp_up_bias": layer.mlp.dense_h_to_4h.bias.data.half(),
            "mlp_down_weight": layer.mlp.dense_4h_to_h.weight.data.half(),
            "mlp_down_bias": layer.mlp.dense_4h_to_h.bias.data.half(),
        }
        all_weights.append(weights)

        k = past_key_values[layer_idx][0].squeeze(0).transpose(0, 1).contiguous()
        v = past_key_values[layer_idx][1].squeeze(0).transpose(0, 1).contiguous()
        k = k.reshape(k.shape[0], -1)
        v = v.reshape(v.shape[0], -1)

        k_cache_full = torch.zeros((max_seq_len, hidden_size), dtype=torch.float16, device=device)
        v_cache_full = torch.zeros((max_seq_len, hidden_size), dtype=torch.float16, device=device)
        k_cache_full[: k.shape[0]] = k
        v_cache_full[: v.shape[0]] = v
        kv_caches.append((k_cache_full, v_cache_full, k.shape[0]))

    torch.cuda.synchronize()
    setup_time = time.time() - start
    return {
        "input_ids": input_ids,
        "prompt_length": prompt_length,
        "first_token": first_token,
        "all_weights": all_weights,
        "kv_caches": kv_caches,
        "setup_time": setup_time,
    }


def decode_clusterfusion(model, prompt, num_new_tokens, state):
    device = next(model.parameters()).device
    num_layers = len(model.gpt_neox.layers)
    num_heads = 32
    head_dim = 80
    hidden_size = 2560
    rotary_dim = 20

    next_token = state["first_token"]
    generated_ids = [next_token.item()]
    input_ids = state["input_ids"]
    prompt_length = state["prompt_length"]
    all_weights = state["all_weights"]
    kv_caches = [
        (k.clone(), v.clone(), cur_len)
        for (k, v, cur_len) in state["kv_caches"]
    ]

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for step in range(num_new_tokens - 1):
            current_position = prompt_length + step
            hidden_states = model.gpt_neox.embed_in(next_token).half().squeeze(1)
            cos, sin = compute_rope_embeddings(current_position, rotary_dim, head_dim, device=device)

            for layer_idx in range(num_layers):
                weights = all_weights[layer_idx]
                k_cache_full, v_cache_full, current_len = kv_caches[layer_idx]

                hidden_states, _, _ = clusterfusion.pythia_decoder_layer(
                    hidden_states,
                    weights["qkv_weight"],
                    weights["qkv_bias"],
                    weights["o_weight"],
                    weights["o_bias"],
                    k_cache_full,
                    v_cache_full,
                    weights["ln_weight"],
                    weights["ln_bias"],
                    cos,
                    sin,
                    weights["post_ln_weight"],
                    weights["post_ln_bias"],
                    weights["mlp_up_weight"],
                    weights["mlp_up_bias"],
                    weights["mlp_down_weight"],
                    weights["mlp_down_bias"],
                    current_len,
                )
                kv_caches[layer_idx] = (k_cache_full, v_cache_full, current_len + 1)

            hidden_states = torch.nn.functional.layer_norm(
                hidden_states,
                (hidden_size,),
                model.gpt_neox.final_layer_norm.weight.data,
                model.gpt_neox.final_layer_norm.bias.data,
                eps=1e-5,
            )
            logits = model.embed_out(hidden_states)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

    torch.cuda.synchronize()
    decode_time = time.time() - start
    return decode_time, generated_ids


def decode_hf(model, input_ids, num_new_tokens):
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=num_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    decode_time = time.time() - start
    return decode_time, output_ids[0].tolist()


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    results = []
    for num_tokens in TOKEN_COUNTS:
        state = prepare_setup(model, tokenizer, PROMPT, num_tokens)
        cluster_time, ids_kernel = decode_clusterfusion(model, PROMPT, num_tokens, state)
        hf_time, ids_hf = decode_hf(model, state["input_ids"], num_tokens)

        results.append(
            {
                "tokens": num_tokens,
                "cf_decode_s": cluster_time,
                "hf_decode_s": hf_time,
                "speedup": hf_time / cluster_time if cluster_time > 0 else float("inf"),
                "match": ids_hf == (state["input_ids"][0].tolist() + ids_kernel),
                "setup_s": state["setup_time"],
            }
        )

    print("\n=== Decode Time (excluding setup) ===")
    header = f"{'tokens':>8} | {'CF_decode(s)':>12} | {'HF_decode(s)':>12} | {'speedup':>8} | {'match':>6} | {'setup_excl?':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['tokens']:8d} | {r['cf_decode_s']:12.3f} | {r['hf_decode_s']:12.3f} | {r['speedup']:8.2f} | {str(r['match']):>6} | {'excluded':>12}"
        )


if __name__ == "__main__":
    main()


