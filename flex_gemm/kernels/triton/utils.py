from typing import *
import torch
import triton


def get_gpu_name():
    return torch.cuda.get_device_name()


def get_platform_name():
    name = get_gpu_name()
    if 'NVIDIA' in name:
        return 'cuda'
    elif 'AMD' in name:
        return 'hip'
    else:
        return 'unknown'
    

def get_num_sm():
    return torch.cuda.get_device_properties("cuda").multi_processor_count
    

def get_autotune_config(
    default: List[triton.Config] = None,
    platform: Dict[str, List[triton.Config]] = None,
    device: Dict[str, List[triton.Config]] = None,
) -> List[triton.Config]:
    """
    Get the autotune configuration for the current platform and device.
    """
    if device is not None:
        gpu_name = get_gpu_name()
        for key, value in device.items():
            if key.lower() in gpu_name.lower():
                return value
    
    if platform is not None:
        platform_name = get_platform_name()
        for key, value in platform.items():
            if key.lower() in platform_name.lower():
                return value
    
    if default is None:
        raise ValueError("No autotune configuration found for the current platform and device.")
    return default
