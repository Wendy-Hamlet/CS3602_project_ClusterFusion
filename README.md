# CS3602 Project: ClusterFusion for Pythia-2.8B

This project implements a CUDA-accelerated decoder layer for EleutherAI Pythia-2.8B model, focusing on the **Attention + MLP Up + GELU** computation path.

## Supported Model

| Model | Architecture | Status |
|-------|--------------|--------|
| **Pythia-2.8B** | GPT-NeoX | ✅ Optimized |

## What's Accelerated

The following operations are fused into a single CUDA kernel:

| Operation | Status | Description |
|-----------|--------|-------------|
| LayerNorm | ✅ CUDA | Pre-attention normalization |
| QKV Projection | ✅ CUDA | Query, Key, Value computation |
| RoPE | ✅ CUDA | Rotary Position Embedding |
| Flash Decoding | ✅ CUDA | Memory-efficient attention |
| Output Projection | ✅ CUDA | Attention output |
| Post-LayerNorm | ✅ CUDA | Pre-MLP normalization |
| MLP Up + GELU | ✅ CUDA | First MLP layer with activation |
| MLP Down | PyTorch | Second MLP layer (simple matmul) |

## Performance Results

Benchmarked on NVIDIA RTX 5090 (sm_120), batch=1, seq_len=64:

### Per-Layer Benchmark (Attention + MLP Up + GELU)

| Metric | PyTorch | ClusterFusion | Speedup |
|--------|---------|---------------|---------|
| Time per layer | 0.27 ms | 0.14 ms | **1.88x** |
| 32 layers total | 8.69 ms | 4.61 ms | **1.88x** |

### Why This Kernel is Fast

The 1.88x speedup comes from fusing multiple operations:
- **7 operations fused**: LayerNorm → QKV → RoPE → Attention → Output → PostLN → MLP Up → GELU
- **Single kernel launch**: Eliminates 6+ kernel launch overheads
- **Register/shared memory reuse**: Intermediate data stays on-chip
- **TMA acceleration**: Hardware-accelerated weight loading

## Environment

- Python 3.13 (conda), NVIDIA GPU with `sm_120` compute capability
- CUDA 12.8+ user-space wheels via PyTorch cu130 index

## Quick Start

```bash
# Create environment
conda create -n nlp_project python=3.13 -y
conda activate nlp_project

# Core DL stack (cu130 wheels)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Kernel + HF stack
pip install flashinfer-python
pip install transformers accelerate datasets

# ClusterFusion build
pip install -e .

# Test (use HF mirror for model download if needed)
export HF_ENDPOINT=https://hf-mirror.com
python tests/test_attention_only.py
```

## API Usage

```python
import clusterfusion
import torch.nn.functional as F

# CUDA kernel: Attention + MLP Up + GELU
attn_output, mlp_intermediate, k_new, v_new = clusterfusion.pythia_2b8_attention_only(
    hidden_states,          # [1, 1, 2560]
    qkv_weight, qkv_bias,   # QKV projection
    o_weight, o_bias,       # Output projection
    k_cache, v_cache,       # KV cache
    ln_weight, ln_bias,     # LayerNorm
    cos, sin,               # RoPE embeddings
    post_ln_weight, post_ln_bias,
    mlp_up_weight, mlp_up_bias,
    current_seq_len
)

# Complete the layer with PyTorch MLP Down
mlp_output = F.linear(mlp_intermediate, mlp_down_weight, mlp_down_bias)
output = hidden_states + attn_output + mlp_output  # Parallel residual
```

## Key Optimizations

1. **Kernel Fusion**: All attention operations in a single kernel launch
2. **TMA Weight Loading**: Hardware-accelerated tensor memory access
3. **Flash Decoding**: Online softmax with numerical stability
4. **Cluster-level Reduction**: Efficient cross-block communication
5. **PTX GELU**: Fast approximation using hardware intrinsics

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Hidden Size | 2560 |
| Attention Heads | 32 |
| Head Dimension | 80 |
| FFN Dimension | 10240 |
| Layers | 32 |

## Files

| File | Description |
|------|-------------|
| `include/5090/pythia_2b8/kernel_attention.cuh` | CUDA kernel implementation |
| `include/5090/pythia_2b8/pythia_attention_dispatch.cu` | Kernel dispatch |
| `tests/test_attention_only.py` | Test and benchmark |

## Requirements

- Python 3.13+ (conda recommended)
- PyTorch 2.0+ with CUDA (cu130 wheels)
- NVIDIA GPU with `sm_120` compute capability (RTX 5090 / Blackwell)
- CUDA 12.8+
- flashinfer-python

## Citation

```bibtex
@misc{luo2025clusterfusion,
      title={ClusterFusion: Expanding Operator Fusion Scope for LLM Inference},
      author={Xinhao Luo et al.},
      year={2025},
      eprint={2508.18850},
      archivePrefix={arXiv}
}
```
