# CS3602 Project: ClusterFusion for Pythia-2.8B

This project implements a CUDA-accelerated decoder layer for EleutherAI Pythia-2.8B model, focusing on the **Attention + MLP Up + GELU** computation path.

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
python tests/test_pythia.py
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

## Files

| File | Description |
|------|-------------|
| `include/5090/pythia_2b8/kernel_attention.cuh` | CUDA kernel implementation |
| `include/5090/pythia_2b8/pythia_attention_dispatch.cu` | Kernel dispatch |
| `tests/test_pythia.py` | Correctness test and per-layer benchmark |
| `tests/benchmark_e2e.py` | End-to-end TPOT benchmark |

## Requirements

- Python 3.13+ (conda recommended)
- PyTorch 2.0+ with CUDA (cu130 wheels)
- NVIDIA GPU with `sm_120` compute capability (RTX 5090 / Blackwell)
- CUDA 12.8+
- flashinfer-python

## Acknowledgments

This implementation is inspired by the ClusterFusion framework, which introduces cluster-level operator fusion for LLM inference on NVIDIA Blackwell GPUs. For more details, please refer to:

**ClusterFusion: Expanding Operator Fusion Scope for LLM Inference**  
Xinhao Luo et al., 2025  
arXiv: [2508.18850](https://arxiv.org/abs/2508.18850)

We extend their work by focusing specifically on the Attention + MLP Up computation path for Pythia-2.8B, demonstrating the performance benefits of kernel fusion on decode-time inference.
