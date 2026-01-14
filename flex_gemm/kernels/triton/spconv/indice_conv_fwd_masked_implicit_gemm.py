from typing import *
import math
import torch
import triton
import triton.language as tl
from ....utils.autotuner import triton_autotune
from . import config


heuristics = {
    'valid_kernel': lambda args: args['valid_kernel'](args['B1']),
    'valid_kernel_seg': lambda args: args['valid_kernel_seg'](args['B1']),
}


@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'LOGM', 'Ci', 'Co', 'V', 'allow_tf32'],
)
@triton.heuristics(heuristics)
@triton.jit
def indice_conv_fwd_masked_implicit_gemm_kernel(
    input,
    weight,
    bias,
    neighbor,
    sorted_idx,
    output,
    # Tensor dimensions
    M, LOGN, LOGM, Ci, Co, V: tl.constexpr,
    # Meta-parameters
    B1: tl.constexpr,   # Block size for M dimension
    B2: tl.constexpr,   # Block size for Co dimension
    BK: tl.constexpr,   # Block size for K dimension (V * Ci)
    allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
    # Huristic parameters
    valid_kernel,
    valid_kernel_seg,
):
    """
    Indice convolution forward kernel using masked implicit GEMM.
    
    Args:
        input (pointer): A pointer to the input tensor of shape (N, Ci)
        weight (pointer): A pointer to the weight tensor of shape (Co, V, Ci)
        bias (pointer): A pointer to the bias tensor of shape (Co)
        neighbor (pointer): A pointer to the neighbor tensor of shape (M, V)
        sorted_idx (pointer): A pointer to the sorted index tensor of shape (M)
        valid_kernel (pointer): A pointer to the valid neighbor index tensor of shape (L)
        valid_kernel_seg (pointer): A pointer to the valid neighbor index segment tensor of shape (BLOCK_M + 1)
        output (pointer): A pointer to the output tensor of shape (M, Co)
    """
    block_id = tl.program_id(axis=0)
    block_dim_co = tl.cdiv(Co, B2)
    block_id_co = block_id % block_dim_co
    block_id_m = block_id // block_dim_co
    
    # Create pointers for submatrices of A and B.
    num_k = tl.cdiv(Ci, BK)  # Number of blocks in K dimension
    valid_kernel_start = tl.load(valid_kernel_seg + block_id_m)
    valid_kernel_seglen = tl.load(valid_kernel_seg + block_id_m + 1) - valid_kernel_start
    offset_m = block_id_m * B1 + tl.arange(0, B1)
    m_mask = offset_m < M
    offset_sorted_m = tl.load(sorted_idx + offset_m, mask=m_mask, other=0)  # (B1,)
    offset_co = (block_id_co * B2 + tl.arange(0, B2)) % Co                  # (B2,)
    offset_k = tl.arange(0, BK)                                             # (BK,)
    
    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, B2), dtype=tl.float32)
    
    # Iterate along V*Ci dimension.
    for k in range(num_k * valid_kernel_seglen):
        v = k // num_k
        bk = k % num_k
        v = tl.load(valid_kernel + valid_kernel_start + v)
        # Calculate pointers to input matrix.
        neighbor_offset = tl.load(neighbor + offset_sorted_m * V + v)                             # (B1,)
        input_ptr = input + bk * BK + (neighbor_offset[:, None].to(tl.int64) * Ci + offset_k[None, :])         # (B1, BK)
        # Calculate pointers to weight matrix.
        weight_ptr = weight + v * Ci + bk * BK + (offset_co[None, :] * V * Ci + offset_k[:, None])  # (BK, B2)
        # Load the next block of input and weight.
        neigh_mask = neighbor_offset != 0xffffffff
        k_mask = offset_k < Ci - bk * BK
        input_block = tl.load(input_ptr, mask=neigh_mask[:, None] & k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptr, mask=k_mask[:, None], other=0.0)
        # Accumulate along the K dimension.
        accumulator = tl.dot(input_block, weight_block, accumulator,
                             input_precision='tf32' if allow_tf32 else 'ieee')                      # (B1, B2)
    c = accumulator.to(input.type.element_ty)
            
    # add bias
    if bias is not None:
        bias_block = tl.load(bias + offset_co)
        c += bias_block[None, :]
                
    # Write back the block of the output matrix with masks.
    out_offset_m = offset_sorted_m
    out_offset_co = block_id_co * B2 + tl.arange(0, B2)
    out_ptr = output + (out_offset_m[:, None] * Co + out_offset_co[None, :])
    out_mask = m_mask[:, None] & (out_offset_co[None, :] < Co)
    tl.store(out_ptr, c, mask=out_mask)


def indice_conv_fwd_masked_implicit_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    neighbor: torch.Tensor,
    sorted_idx: torch.Tensor,
    valid_kernel: Callable[[int], torch.Tensor],
    valid_kernel_seg: Callable[[int], torch.Tensor],
) -> torch.Tensor:
    assert input.shape[1] == weight.shape[2], "Incompatible dimensions"
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert neighbor.is_contiguous(), "Matrix neighbor must be contiguous"
    N, M, Ci, Co, V = input.shape[0], neighbor.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
    LOGN = int(math.log2(N))
    LOGM = int(math.log2(M))
    # Allocate output matrix output.
    output = torch.empty((M, Co), device=input.device, dtype=input.dtype)
    # Launch the kernel.
    grid = lambda META: (triton.cdiv(Co, META['B2']) * triton.cdiv(M, META['B1']),)
    indice_conv_fwd_masked_implicit_gemm_kernel[grid](
        input, weight, bias, neighbor, sorted_idx, output,
        M, LOGN, LOGM, Ci, Co, V,
        valid_kernel=valid_kernel,
        valid_kernel_seg=valid_kernel_seg,
        allow_tf32=config.allow_tf32,
    )
    return output
