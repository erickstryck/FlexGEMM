template<typename T>
__forceinline__ __device__ T hash(T k, size_t N);

// 32 bit Murmur3 hash
template<>
__forceinline__ __device__ uint32_t hash<uint32_t>(uint32_t k, size_t N) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k % N;
}


// 64 bit Murmur3 hash
template<>
__forceinline__ __device__ uint64_t hash<uint64_t>(uint64_t k, size_t N) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k % N;
}


template<typename T>
__forceinline__ __device__ void linear_probing_insert(
    T* hashmap,
    const T key,
    const T values,
    const size_t N
) {
    T slot = hash(key, N);
    while (true) {
        T prev = atomicCAS(&hashmap[slot], std::numeric_limits<T>::max(), key);
        if (prev == std::numeric_limits<T>::max() || prev == key) {
            hashmap[slot + N] = values;
            return;
        }
        slot = slot + 1;
        if (slot >= N) slot = 0;
    }
}


template<>
__forceinline__ __device__ void linear_probing_insert<uint64_t>(
    uint64_t* hashmap,
    const uint64_t key,
    const uint64_t value,
    const size_t N
) {
    uint64_t slot = hash(key, N);
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long*>(&hashmap[slot]),
            static_cast<unsigned long long>(std::numeric_limits<uint64_t>::max()),
            static_cast<unsigned long long>(key)
        );
        if (prev == std::numeric_limits<uint64_t>::max() || prev == key) {
            hashmap[slot + N] = value;
            return;
        }
        slot = (slot + 1) % N;
    }
}


template<typename T>
__forceinline__ __device__ uint32_t linear_probing_lookup(
    const T* hashmap,
    const T key,
    const size_t N
) {
    T slot = hash(key, N);
    while (true) {
        T prev = hashmap[slot];
        if (prev == std::numeric_limits<T>::max()) {
            return std::numeric_limits<T>::max();
        }
        if (prev == key) {
            return hashmap[slot + N];
        }
        slot = slot + 1;
        if (slot >= N) slot = 0;
    }
}
