import math
import torch
import triton
import triton.language as tl
from ..utils import get_num_sm
from ....utils.autotuner import triton_autotune, autotune
from . import config
from .indice_conv_fwd_implicit_gemm import indice_conv_fwd_implicit_gemm_kernel


@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'LOGM', 'Ci', 'Co', 'V', 'SPLITK', 'allow_tf32'],
)
@triton.jit
def indice_conv_fwd_implicit_gemm_splitk_kernel(
    input,
    weight,
    bias,
    neighbor,
    output,
    # Tensor dimensions
    M, LOGN, LOGM, Ci, Co, V: tl.constexpr,
    # Meta-parameters
    B1: tl.constexpr,   # Block size for M dimension
    B2: tl.constexpr,   # Block size for Co dimension
    BK: tl.constexpr,   # Block size for K dimension (V * Ci)
    SPLITK: tl.constexpr,  # Split K dimension
    allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
):
    """
    Indice convolution forward kernel using implicit GEMM with split K dimension.
    
    Args:
        input (pointer): A pointer to the input tensor of shape (N, Ci)
        weight (pointer): A pointer to the weight tensor of shape (Co, V, Ci)
        bias (pointer): A pointer to the bias tensor of shape (Co)
        neighbor (pointer): A pointer to the neighbor tensor of shape (M, V)
        output (pointer): A pointer to the output tensor of shape (M, Co)
    """
    block_id_k = tl.program_id(axis=1)  # SplitK dimension
    block_id = tl.program_id(axis=0)
    block_dim_co = tl.cdiv(Co, B2)
    block_id_co = block_id % block_dim_co
    block_id_m = block_id // block_dim_co
    
    # Create pointers for submatrices of A and B.
    num_k = tl.cdiv(Ci, BK)  # Number of blocks in K dimension
    k_start = tl.cdiv(num_k * V * block_id_k, SPLITK)
    k_end = tl.cdiv(num_k * V * (block_id_k + 1), SPLITK)
    offset_m = (block_id_m * B1 + tl.arange(0, B1)) % M         # (B1,)
    offset_co = (block_id_co * B2 + tl.arange(0, B2)) % Co      # (B2,)
    offset_k = tl.arange(0, BK)                                 # (BK,)
    
    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, B2), dtype=tl.float32)
    
    # Calculate pointers to weight matrix.
    weight_ptr = weight + k_start * BK + (offset_co[None, :] * V * Ci + offset_k[:, None])     # (BK, B2)
    
    # Iterate along V*Ci dimension.
    for k in range(k_start, k_end):
        v = k // num_k
        bk = k % num_k
        # Calculate pointers to input matrix.
        neighbor_offset = tl.load(neighbor + offset_m * V + v)                                # (B1,)
        input_ptr = input + bk * BK + (neighbor_offset[:, None].to(tl.int64) * Ci + offset_k[None, :])     # (B1, BK)
        # Load the next block of input and weight.
        neigh_mask = neighbor_offset != 0xffffffff
        k_mask = offset_k < Ci - bk * BK
        input_block = tl.load(input_ptr, mask=neigh_mask[:, None] & k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptr, mask=k_mask[:, None], other=0.0)
        # Accumulate along the K dimension.
        accumulator = tl.dot(input_block, weight_block, accumulator,
                             input_precision='tf32' if allow_tf32 else 'ieee')                  # (B1, B2)
        # Advance the pointers to the next Ci block.
        weight_ptr += min(BK, Ci - bk * BK)
            
    # add bias
    if bias is not None and block_id_k == 0:
        bias_block = tl.load(bias + offset_co)
        accumulator += bias_block[None, :]
                
    # Write back the block of the output matrix with masks.
    out_offset_m = block_id_m * B1 + tl.arange(0, B1)
    out_offset_co = block_id_co * B2 + tl.arange(0, B2)
    out_ptr = output + block_id_k * M * Co + (out_offset_m[:, None] * Co + out_offset_co[None, :])
    out_mask = (out_offset_m[:, None] < M) & (out_offset_co[None, :] < Co)
    tl.store(out_ptr, accumulator, mask=out_mask)


def indice_conv_fwd_implicit_gemm_splitk_configs(input, weight, bias, neighbor):
    M, Co = neighbor.shape[0], weight.shape[0]
    MAX_NB1 = (M + 128 - 1) // 128
    MAX_NB2 = (Co + 128 - 1) // 128
    NUM_BLOCKS = MAX_NB1 * MAX_NB2
    MIN_NUM_BLOCKS = get_num_sm()
    MAX_NUM_BLOCKS = 32 * get_num_sm()
    MIN_NUM_BLOCKS_LOG2 = max(0, int(math.log2(MIN_NUM_BLOCKS / NUM_BLOCKS)))
    MAX_NUM_BLOCKS_LOG2 = max(1, int(math.log2(MAX_NUM_BLOCKS / NUM_BLOCKS) + 1))
    configs = []
    for i in range(MIN_NUM_BLOCKS_LOG2, MAX_NUM_BLOCKS_LOG2):
        configs.append({'SPLITK': 2 ** i})
    return configs


def indice_conv_fwd_implicit_gemm_splitk_keys(input, weight, bias, neighbor):
    N, M, Ci, Co, V = input.shape[0], neighbor.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
    return f'(2^{int(math.log2(N))}, 2^{int(math.log2(M))}, {Ci}, {Co}, {V})'


@autotune(
    config_fn=indice_conv_fwd_implicit_gemm_splitk_configs,
    key_fn=indice_conv_fwd_implicit_gemm_splitk_keys,
)
def indice_conv_fwd_implicit_gemm_splitk(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    neighbor: torch.Tensor,
    SPLITK: int = 1,
) -> torch.Tensor:
    assert input.shape[1] == weight.shape[2], "Incompatible dimensions"
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert neighbor.is_contiguous(), "Matrix neighbor must be contiguous"
    N, M, Ci, Co, V = input.shape[0], neighbor.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
    LOGN = int(math.log2(N))
    LOGM = int(math.log2(M))
    # Launch the kernel.
    if SPLITK == 1:
        output = torch.empty((M, Co), device=input.device, dtype=input.dtype)
        grid = lambda META: (triton.cdiv(Co, META['B2']) * triton.cdiv(M, META['B1']),)
        indice_conv_fwd_implicit_gemm_kernel[grid](
            input, weight, bias, neighbor, output,
            M, LOGN, LOGM, Ci, Co, V,
            allow_tf32=config.allow_tf32,
        )
        return output
    else:
        output = torch.empty((SPLITK, M, Co), device=input.device, dtype=torch.float32)
        grid = lambda META: (triton.cdiv(Co, META['B2']) * triton.cdiv(M, META['B1']), SPLITK)
        indice_conv_fwd_implicit_gemm_splitk_kernel[grid](
            input, weight, bias, neighbor, output,
            M, LOGN, LOGM, Ci, Co, V,
            SPLITK=SPLITK,
            allow_tf32=config.allow_tf32,
        )
        return output.sum(dim=0).to(input.dtype)
