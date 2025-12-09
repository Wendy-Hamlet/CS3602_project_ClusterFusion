import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the custom kernel
import clusterfusion

def generate_random_weights(shape, seed=42):
    """Generate random weights with a fixed seed for reproducibility."""
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float16, device='cuda')

def test_qkv_projection():
    """Test only the QKV projection part to isolate the issue."""
    
    # Model configuration
    hidden_size = 2560
    num_heads = 32
    head_dim = 80
    
    print("=== Testing QKV Projection Only ===")
    print(f"Hidden size: {hidden_size}")
    print(f"Num heads: {num_heads}")
    print(f"Head dim: {head_dim}")
    print()
    
    # Generate test data
    input_hidden = generate_random_weights((1, hidden_size))
    weight_qkv = generate_random_weights((3 * num_heads * head_dim, hidden_size))
    layernorm_weight = generate_random_weights((hidden_size,))
    
    # Normalize input (RMSNorm)
    eps = 1e-5
    variance = input_hidden.pow(2).mean(-1, keepdim=True)
    hidden_normed = input_hidden * torch.rsqrt(variance + eps) * layernorm_weight
    
    print("Input after normalization (first 20, last 20):")
    print(hidden_normed[0, :20])
    print(hidden_normed[0, -20:])
    print()
    
    # Reference: Naive QKV projection
    print("--- Reference (Naive PyTorch) ---")
    qkv_ref = torch.matmul(hidden_normed, weight_qkv.t())  # (1, 7680)
    q_ref = qkv_ref[:, :hidden_size].reshape(1, num_heads, head_dim)  # (1, 32, 80)
    k_ref = qkv_ref[:, hidden_size:2*hidden_size].reshape(1, num_heads, head_dim)
    v_ref = qkv_ref[:, 2*hidden_size:].reshape(1, num_heads, head_dim)
    
    print(f"Q shape: {q_ref.shape}")
    print(f"Q[0, 0, :20] (head 0, first 20 dims): {q_ref[0, 0, :20]}")
    print(f"Q[0, 0, -20:] (head 0, last 20 dims): {q_ref[0, 0, -20:]}")
    print()
    print(f"Q[0, 1, :20] (head 1, first 20 dims): {q_ref[0, 1, :20]}")
    print(f"Q[0, 1, -20:] (head 1, last 20 dims): {q_ref[0, 1, -20:]}")
    print()
    
    print(f"K shape: {k_ref.shape}")
    print(f"K[0, 0, :20] (head 0, first 20 dims): {k_ref[0, 0, :20]}")
    print(f"K[0, 1, :20] (head 1, first 20 dims): {k_ref[0, 1, :20]}")
    print()
    
    print(f"V shape: {v_ref.shape}")
    print(f"V[0, 0, :20] (head 0, first 20 dims): {v_ref[0, 0, :20]}")
    print()
    
    # Now test with manual loop to understand indexing
    print("--- Manual Loop (Understanding Indexing) ---")
    # Simulate what the kernel should do
    # For head_id = 1 (second head), cluster_block_id = 0 (first block covering dims 0-127)
    
    head_id = 1
    block_size = 128  # DIM_PER_BLOCK
    block_start = 0
    block_end = block_size
    
    # Extract the part of normalized input for this block
    block_input = hidden_normed[0, block_start:block_end]  # (128,)
    
    # Extract weight for Q projection of this head
    head_start = head_id * head_dim  # 80
    head_end = (head_id + 1) * head_dim  # 160
    
    # Weight shape: (7680, 2560) = (3*32*80, 2560)
    # For Q: rows 0:2560 (all heads Q concatenated)
    # For head 1 Q: rows 80:160
    weight_q_head1 = weight_qkv[head_start:head_end, block_start:block_end]  # (80, 128)
    
    # Manual matmul: Q = input @ weight.T
    # input: (128,), weight_q_head1: (80, 128)
    # Q = input @ weight_q_head1.T = (128,) @ (128, 80) = (80,)
    q_manual = torch.matmul(block_input, weight_q_head1.t())  # (80,)
    
    print(f"Manual Q for head 1 (using block 0 input): {q_manual[:20]}")
    print(f"Reference Q for head 1: {q_ref[0, 1, :20]}")
    print(f"Difference: {(q_manual[:20] - q_ref[0, 1, :20]).abs().max().item()}")
    print()
    
    # Now check the kernel's indexing logic
    print("--- Analyzing Kernel Indexing ---")
    print("Kernel uses:")
    print("  weight[(input_idx + i + d) * HEAD_DIM + weight_idx]")
    print("  where:")
    print("    input_idx: position in input (0-127 for block 0)")
    print("    weight_idx: position in output (0-79 for this head)")
    print()
    
    # Simulate kernel indexing
    # The weight is stored as [HEAD_DIM, HIDDEN_DIM] in shared memory after TMA load
    # Actually, looking at the dispatch code:
    # uint64_t size[rank] = {HIDDEN_DIM, 3 * HIDDEN_DIM};
    # This means dim0=HIDDEN_DIM=2560, dim1=3*HIDDEN_DIM=7680
    # So the tensor is loaded with dimensions [2560, 7680]
    # But we want to access Q weights for head 1, which are rows 80:160 of the weight matrix
    
    # Let me check what cluster_head_idx and cluster_block_st_id mean:
    # cluster_head_idx = head_id * HEAD_DIM = 1 * 80 = 80
    # cluster_block_st_id = cluster_block_id * DIM_PER_BLOCK = 0 * 128 = 0
    # TMA loads: cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, cluster_head_idx=80, cluster_block_st_id=0)
    
    print("Tensor map configuration:")
    print("  size[rank] = {HIDDEN_DIM=2560, 3*HIDDEN_DIM=7680}")
    print("  stride[rank-1] = {HIDDEN_DIM * sizeof(half) = 2560 * 2}")
    print()
    print("For head_id=1, cluster_block_id=0:")
    print("  cluster_head_idx = 1 * 80 = 80 (selects rows 80:160 of weight matrix)")
    print("  cluster_block_st_id = 0 * 128 = 0 (selects columns 0:128 of weight matrix)")
    print()
    print("So TMA should load weight_qkv[80:160, 0:128] for Q projection")
    print("But kernel accesses: weight[(input_idx + i + d) * HEAD_DIM + weight_idx]")
    print("This assumes weight is in [BLOCK_SIZE, HEAD_DIM] layout in shared memory")
    print()
    
    # Verify the actual memory layout
    print("=== Memory Layout Analysis ===")
    print("If TMA loads correctly:")
    print("  Shared memory should contain weight_qkv[80:160, 0:128].T")
    print("  i.e., transposed to [128, 80] layout")
    print()
    print("Then kernel access weight[input_idx * HEAD_DIM + weight_idx]")
    print("  gives weight[i * 80 + j] which is correct for [128, 80] layout")
    print()

if __name__ == "__main__":
    test_qkv_projection()
