import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tqdm import tqdm
import torch
import flex_gemm
from flex_gemm.ops.spconv import SubMConv3dFunction
from utils import sphere_coords, benchmark_kernel

    
def egemm_torch_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, **kwargs):
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    dilation = (1, 1, 1)
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.EXPLICIT_GEMM)
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache_torch(coords, shape, ksize, dilation)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }
    

def egemm_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, **kwargs):
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    dilation = (1, 1, 1)
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.EXPLICIT_GEMM)
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }
    
    
def igemm_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, **kwargs):
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    dilation = (1, 1, 1)
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.IMPLICIT_GEMM)
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }
    

def igemmk_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, **kwargs):
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    dilation = (1, 1, 1)
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.IMPLICIT_GEMM_SPLITK)
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }
    

def migemm_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, **kwargs):
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    dilation = (1, 1, 1)
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM)
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }
    

def migemmk_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, **kwargs):
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    dilation = (1, 1, 1)
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }


def test_neighbor_cache():
    # Matrix dimensions.
    config = [
        {'RES': 8, 'C': 1024},
        {'RES': 16, 'C': 1024},
        {'RES': 32, 'C': 1024},
        {'RES': 64, 'C': 1024},
        {'RES': 128, 'C': 512},
        {'RES': 256, 'C': 256},
        {'RES': 512, 'C': 128},
        # {'RES': 1024, 'C': 64},
        # {'RES': 2048, 'C': 32},
    ]
    
    # List of custom kernel functions.
    kernel_functions = {
        'egemm_torch': (egemm_torch_prepare_fn, None),
        'egemm': (egemm_prepare_fn, None),
        'igemm': (igemm_prepare_fn, None),
        'igemmk': (igemmk_prepare_fn, None),
        'migemm': (migemm_prepare_fn, None),
        'migemmk': (migemmk_prepare_fn, None),
    }
    
    results = {}
    for c in tqdm(config, leave=False):
        RES, C = c['RES'], c['C']

        # Create random input matrices.
        feats, coords, shape = sphere_coords(RES, C, dtype=torch.float16)
        weight = torch.randn(C, 3, 3, 3, C, device=feats.device, dtype=feats.dtype)
        args = {
            'coords': coords,
            'shape': shape,
            'weight': weight,
        }

        config_key = f'RES={RES},C={C}'
        results[config_key] = []

        C_ref = egemm_torch_prepare_fn(**args)['neighbor_cache'].neighbor_map

        # Benchmark each custom kernel.
        for kernel_fn, prepare_fn in kernel_functions.values():
            avg_time, memory, C_kernel = benchmark_kernel(kernel_fn, **args, prepare_fn=prepare_fn)
            C_kernel = C_kernel['neighbor_cache'].neighbor_map
            assert torch.equal(C_kernel, C_ref), f"Neighbor cache mismatch for {kernel_fn.__name__}."
            results[config_key].append(f'{avg_time:.3f}/{memory:.3f}G')

    # Print results as a formatted table.
    print("\nSubMConv Neighbor Cache Benchmark Results")
    print("-" * 180)
    items = [f'{"settings":<15}']
    for f in kernel_functions.keys():
        items.append(f'{f:<20}')
    print(' | '.join(items))
    print("-" * 180)
    for k, v in results.items():
        items = [f'{k:<15}']
        items.extend([f'{x:<20}' for x in v])
        print(' | '.join(items))
        

if __name__ == "__main__":
    test_neighbor_cache()
