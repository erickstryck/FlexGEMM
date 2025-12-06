#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "api.h"
#include "hash.cuh"


template<typename T>
static __global__ void hashmap_insert_cuda_kernel(
    const size_t N,
    const size_t M,
    T* __restrict__ hashmap,
    const T* __restrict__ keys,
    const T* __restrict__ values
) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M)
    {
        T key = keys[thread_id];
        T value = values[thread_id];
        linear_probing_insert(hashmap, key, value, N);
    }
}


/**
 * Insert keys into the hashmap
 * 
 * @param hashmap   [2N] uint32/uint64 tensor containing the hashmap (key-value pairs)
 * @param keys      [M] uint32/uint64 tensor containing the keys to be inserted
 * @param values    [M] uint32/uint64 tensor containing the values to be inserted
 */
void hashmap_insert_cuda(
    torch::Tensor& hashmap,
    const torch::Tensor& keys,
    const torch::Tensor& values
) {
    // Dispatch to 32-bit or 64-bit kernel
    if (hashmap.dtype() == torch::kUInt32) {
        hashmap_insert_cuda_kernel<<<
            (keys.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            keys.size(0),
            hashmap.data_ptr<uint32_t>(),
            keys.data_ptr<uint32_t>(),
            values.data_ptr<uint32_t>()
        );
    } else if (hashmap.dtype() == torch::kUInt64) {
        hashmap_insert_cuda_kernel<<<
            (keys.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            keys.size(0),
            hashmap.data_ptr<uint64_t>(),
            keys.data_ptr<uint64_t>(),
            values.data_ptr<uint64_t>()
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
}


template<typename T>
static __global__ void hashmap_lookup_cuda_kernel(
    const size_t N,
    const size_t M,
    const T* __restrict__ hashmap,
    const T* __restrict__ keys,
    T* __restrict__ values
) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M) {
        T key = keys[thread_id];
        values[thread_id] = linear_probing_lookup(hashmap, key, N);
    }
}


/**
 * Lookup keys in the hashmap
 * 
 * @param hashmap   [N] uint32/uint64 tensor containing the hashmap (key-value pairs)
 * @param keys      [M] uint32/uint64 tensor containing the keys to be looked up
 * @return          [M] uint32/uint64 tensor containing the values of the keys
 */
torch::Tensor hashmap_lookup_cuda(
    const torch::Tensor& hashmap,
    const torch::Tensor& keys
) {
    // Allocate output tensor
    auto output = torch::empty_like(keys);

    // Dispatch to 32-bit or 64-bit kernel
    if (hashmap.dtype() == torch::kUInt32) {
        hashmap_lookup_cuda_kernel<<<
            (keys.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            keys.size(0),
            hashmap.data_ptr<uint32_t>(),
            keys.data_ptr<uint32_t>(),
            output.data_ptr<uint32_t>()
        );
    } else if (hashmap.dtype() == torch::kUInt64) {
        hashmap_lookup_cuda_kernel<<<
            (keys.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            keys.size(0),
            hashmap.data_ptr<uint64_t>(),
            keys.data_ptr<uint64_t>(),
            output.data_ptr<uint64_t>()
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }

    return output;
}


template<typename T>
static __global__ void hashmap_insert_3d_cuda_kernel(
    const size_t N,
    const size_t M,
    const int W,
    const int H,
    const int D,
    T* __restrict__ hashmap,
    const int32_t* __restrict__ coords,
    const T* __restrict__ values
) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M) {
        int4 coord = reinterpret_cast<const int4*>(coords)[thread_id];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        size_t flat_idx = (size_t)b * W * H * D + (size_t)x * H * D + (size_t)y * D + z;
    T key = static_cast<T>(flat_idx);
        T value = values[thread_id];
        linear_probing_insert(hashmap, key, value, N);
    }
}


/**
 * Insert 3D coordinates into the hashmap
 * 
 * @param hashmap   [2N] uint32/uint64 tensor containing the hashmap (key-value pairs)
 * @param coords    [M, 4] int32 tensor containing the keys to be inserted
 * @param values    [M] uint32/uint64 tensor containing the values to be inserted
 * @param W         the number of width dimensions
 * @param H         the number of height dimensions
 * @param D         the number of depth dimensions
 */
void hashmap_insert_3d_cuda(
    torch::Tensor& hashmap,
    const torch::Tensor& coords,
    const torch::Tensor& values,
    int W,
    int H,
    int D
) {
    // Dispatch to 32-bit or 64-bit kernel
    if (hashmap.dtype() == torch::kUInt32) {
        hashmap_insert_3d_cuda_kernel<<<
            (coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            coords.size(0),
            W,
            H,
            D,
            hashmap.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            values.data_ptr<uint32_t>()
        );
    } else if (hashmap.dtype() == torch::kUInt64) {
        hashmap_insert_3d_cuda_kernel<<<
            (coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            coords.size(0),
            W,
            H,
            D,
            hashmap.data_ptr<uint64_t>(),
            coords.data_ptr<int32_t>(),
            values.data_ptr<uint64_t>()
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
}


template<typename T>
static __global__ void hashmap_lookup_3d_cuda_kernel(
    const size_t N,
    const size_t M,
    const int W,
    const int H,
    const int D,
    const T* __restrict__ hashmap,
    const int32_t* __restrict__ coords,
    T* __restrict__ values
) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M) {
        int4 coord = reinterpret_cast<const int4*>(coords)[thread_id];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
            values[thread_id] = std::numeric_limits<T>::max();
            return;
        }
        size_t flat_idx = (size_t)b * W * H * D + (size_t)x * H * D + (size_t)y * D + z;
    T key = static_cast<T>(flat_idx);
        values[thread_id] = linear_probing_lookup(hashmap, key, N);
    }
}


/**
 * Lookup 3D coordinates in the hashmap
 * 
 * @param hashmap   [2N] uint32/uint64 tensor containing the hashmap (key-value pairs)
 * @param coords    [M, 4] int32 tensor containing the keys to be looked up
 * @param W         the number of width dimensions
 * @param H         the number of height dimensions
 * @param D         the number of depth dimensions
 * 
 * @return          [M] uint32/uint64 tensor containing the values of the keys
 */
torch::Tensor hashmap_lookup_3d_cuda(
    const torch::Tensor& hashmap,
    const torch::Tensor& coords,
    int W,
    int H,
    int D
) {
    // Allocate output tensor
    auto output = torch::empty({coords.size(0)}, torch::dtype(hashmap.dtype()).device(hashmap.device()));

    // Dispatch to 32-bit or 64-bit kernel
    if (hashmap.dtype() == torch::kUInt32) {
        hashmap_lookup_3d_cuda_kernel<<<
            (coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            coords.size(0),
            W,
            H,
            D,
            hashmap.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            output.data_ptr<uint32_t>()
        );
    } else if (hashmap.dtype() == torch::kUInt64) {
        hashmap_lookup_3d_cuda_kernel<<<
            (coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            coords.size(0),
            W,
            H,
            D,
            hashmap.data_ptr<uint64_t>(),
            coords.data_ptr<int32_t>(),
            output.data_ptr<uint64_t>()
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }

    return output;
}


template<typename T>
static __global__ void hashmap_insert_3d_idx_as_val_cuda_kernel(
    const size_t N,
    const size_t M,
    const int W,
    const int H,
    const int D,
    T* __restrict__ hashmap,
    const int32_t* __restrict__ coords
) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M) {
        int4 coord = reinterpret_cast<const int4*>(coords)[thread_id];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        size_t flat_idx = (size_t)b * W * H * D + (size_t)x * H * D + (size_t)y * D + z;
        T key = static_cast<T>(flat_idx);
        linear_probing_insert(hashmap, key, static_cast<T>(thread_id), N);
    }
}


/**
 * Insert 3D coordinates into the hashmap using index as value
 * 
 * @param hashmap   [2N] uint32/uint64 tensor containing the hashmap (key-value pairs)
 * @param coords    [M, 4] int32 tensor containing the keys to be inserted
 * @param W         the number of width dimensions
 * @param H         the number of height dimensions
 * @param D         the number of depth dimensions
 */
void hashmap_insert_3d_idx_as_val_cuda(
    torch::Tensor& hashmap,
    const torch::Tensor& coords,
    int W,
    int H,
    int D
) {
    // Dispatch to 32-bit or 64-bit kernel
    if (hashmap.dtype() == torch::kUInt32) {
        hashmap_insert_3d_idx_as_val_cuda_kernel<<<
            (coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            coords.size(0),
            W,
            H,
            D,
            hashmap.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>()
        );
    } else if (hashmap.dtype() == torch::kUInt64) {
        hashmap_insert_3d_idx_as_val_cuda_kernel<<<
            (coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            coords.size(0),
            W,
            H,
            D,
            hashmap.data_ptr<uint64_t>(),
            coords.data_ptr<int32_t>()
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
}
