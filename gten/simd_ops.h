#pragma once

#include <chrono>
#include <cstdint>
#include <cmath>
#include <iostream>

#include "gten_types.h"



// TODO: Allow AVX without F16C for FP32 mode.
#if defined(__AVX__) && defined(__F16C__)
#define GTEN_SIMD_AVX 1

#include <immintrin.h>

#endif

namespace gten {

namespace ops {

// Number of floats the avx registers (256bit) can process.
#define GTEN_SIMD_VEC_SIZE 8

#ifdef GTEN_SIMD_AVX

// FUNDAMENTAL VECTOR DATA TYPES.
typedef __m256 Vec_f32x8;

// FLOATING POINT VECTOR OPERATIONS

inline Vec_f32x8 vec_f32x8_load(const Float16* src_ptr) {
    return _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u *)(const_cast<Float16*>(src_ptr))));
}

inline Vec_f32x8 vec_f32x8_load(const float* src_ptr) {
    return _mm256_loadu_ps(const_cast<float*>(src_ptr));
}

inline void vec_f32x8_store(Vec_f32x8 vec, float* dest_ptr) {
    _mm256_storeu_ps(dest_ptr, vec);
}

inline void vec_f32x8_store(Vec_f32x8 vec, Float16* dest_ptr) {
    return _mm_storeu_si128((__m128i_u *)dest_ptr, _mm256_cvtps_ph(vec, 0));
}

inline Vec_f32x8 vec_f32x8_add(Vec_f32x8 a, Vec_f32x8 b) {
    return _mm256_add_ps(a, b);
}

inline Vec_f32x8 vec_f32x8_mul(Vec_f32x8 a, Vec_f32x8 b) {
    return _mm256_mul_ps(a, b);
}

// Return A * B + C
inline Vec_f32x8 vec_f32x8_fma(Vec_f32x8 a, Vec_f32x8 b, Vec_f32x8 c) {
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}

inline float vec_f32x8_sum(Vec_f32x8 vec) {
    float* f = (float *)(&vec);
    return f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7];
}

inline Vec_f32x8 vec_f32x8_setzero() {
    return _mm256_setzero_ps();
}

#endif

} // namespace ops

} // namespace gten

