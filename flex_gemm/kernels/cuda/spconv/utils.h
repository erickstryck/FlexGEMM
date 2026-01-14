#pragma once

template<typename T>
inline bool is_safe_mul(T a, T b, T& res) {
    static_assert(std::is_unsigned<T>::value, "is_safe_mul supports unsigned types only.");
    if (a == 0 || b == 0) {
        res = 0;
        return true;
    }
    res = a * b;
    return (res / a == b);
}
