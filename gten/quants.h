#pragma once


#include <cmath>
#include <memory>

#include "gten_types.h"
#include "log.h"

namespace gten {

namespace globs {
static const int q8_block_size = 32;
static const int q4_block_size = 32;
}

struct Q8Block
{
    Float16 delta;
    Qint8 data[globs::q8_block_size];
};

static_assert(sizeof(Q8Block) == sizeof(Float16) + globs::q8_block_size);

struct Q4Block
{
    Float16 delta;
    Qint8 data[globs::q4_block_size / 2];
};

static_assert(sizeof(Q4Block) == sizeof(Float16) + globs::q4_block_size / 2);


namespace ops {

[[nodiscard]]
inline Qint8 q8_quantize_single(float x, float delta) {
    const float id = delta ? 1.0f/delta : 0.0f;

    const float x0 = x * id;
    const Qint8 quantized = static_cast<Qint8>(roundf(x0));

    return quantized;
}


[[nodiscard]]
inline float q8_dequantize_single(Qint8 x, float delta) {
    return x * delta;
}

void q8_quantize_block(const float* inp, Q8Block* out) {
    const int block_size = globs::q8_block_size;

    float absmax = 0;
    for (int j = 0; j < block_size; j++) {
        const float x = inp[j];
        absmax = std::max(absmax, fabsf(x));
    }

    const float delta = absmax / 127.0f;
    out->delta = fp32_to_fp16(delta);

    const float scale = delta ? 1.0f/delta : 0.0f;
    for (int i = 0; i < block_size; i++) {
        out->data[i] = static_cast<Qint8>(roundf(inp[i] * scale));
    }
}


void q8_dequantize_block(const Q8Block* inp, float* out) {
    const int block_size = globs::q8_block_size;

    const float delta = fp16_to_fp32(inp->delta);
    for (int i = 0; i < block_size; i++) {
        out[i] = inp->data[i] * delta;
    }
}

void q4_dequantize_block(const Q4Block* inp, float* out) {
    const int block_size = globs::q4_block_size;

    const float delta = fp16_to_fp32(inp->delta);
    const int half_block_size = block_size / 2;
    for (int i = 0; i < half_block_size; i += 1) {
        const Qint4 packed = inp->data[i];
        const Qint8 high = (packed >> 4) - 7;
        const Qint8 low = (packed & 0b00001111) - 7; 
        out[i] = high * delta;
        out[i+half_block_size] = low * delta;
    }
}

void q8_quantize_row(const float* inp, Q8Block* out, const int rowsize) {
    const int block_size = globs::q8_block_size;
    GTEN_ASSERT(rowsize % block_size == 0);
    const int n_blocks = rowsize / block_size;

    for (int i = 0; i < n_blocks; i++) {
        const float* inp_block_data = inp + i * block_size;
        Q8Block* out_block_data = out + i;

        q8_quantize_block(inp_block_data, out_block_data);
    }
}

void q8_quantize_row_delta(const float* inp, Qint8* out, const float delta, const int rowsize) {
    for (int i = 0; i < rowsize; i++) {
        out[i] = q8_quantize_single(inp[i], delta);
    }
}

void q8_dequantize_row(const Q8Block* inp, float* out, int rowsize) {
    const int block_size = globs::q8_block_size;
    GTEN_ASSERT(rowsize % block_size == 0);
    const int n_blocks = rowsize / block_size;

    for (int i = 0; i < n_blocks; i++) {
        q8_dequantize_block(inp + i, out + i * block_size);
    }
}

void q4_dequantize_row(const Q4Block* inp, float* out, int rowsize) {
    const int block_size = globs::q4_block_size;
    GTEN_ASSERT(rowsize % block_size == 0);
    const int n_blocks = rowsize / block_size;

    for (int i = 0; i < n_blocks; i++) {
        q4_dequantize_block(inp + i, out + i * block_size);
    }
}

inline void q8_dequantize_row_delta(const Qint8* x, float* out, float delta, int size) {
    for (int i = 0; i < size; i++) {
        const Qint8 x_val = x[i];
        out[i] = x_val * delta;
    }
}

} // namespace ops

} // namespace gten