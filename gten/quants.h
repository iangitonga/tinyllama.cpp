#pragma once


#include <cmath>
#include <memory>

#include "gten_types.h"
#include "log.h"

namespace gten {

namespace globs {
static const int q8_block_size = 32;
}

struct Q8Block
{
    Float16 delta;
    Qint8 data[globs::q8_block_size];
};


namespace ops {

[[nodiscard]]
inline Qint8 quantize_single(float x, float delta) {
    const float id = delta ? 1.0f/delta : 0.0f;

    const float x0 = x * id;
    const Qint8 quantized = roundf(x0);

    return quantized;
}


[[nodiscard]]
inline float dequantize_single(Qint8 x, float delta) {
    return x * delta;
}

void quantize_block(const float* inp, Q8Block* out) {
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


void dequantize_block(const Q8Block* inp, float* out) {
    const int block_size = globs::q8_block_size;

    const float delta = fp16_to_fp32(inp->delta);
    for (int i = 0; i < block_size; i++) {
        out[i] = inp->data[i] * delta;
    }
}

void quantize_row(const float* inp, Q8Block* out, const int rowsize) {
    const int block_size = globs::q8_block_size;
    GTEN_ASSERT(rowsize % block_size == 0);
    const int n_blocks = rowsize / block_size;

    for (int i = 0; i < n_blocks; i++) {
        const float* inp_block_data = inp + i * block_size;
        Q8Block* out_block_data = out + i;

        quantize_block(inp_block_data, out_block_data);
    }
}

void dequantize_row(const Q8Block* inp, float* out, int rowsize) {
    const int block_size = globs::q8_block_size;
    GTEN_ASSERT(rowsize % block_size == 0);
    const int n_blocks = rowsize / block_size;

    for (int i = 0; i < n_blocks; i++) {
        dequantize_block(inp + i, out + i * block_size);
    }
}

inline void dequantize_row(const Qint8* x, float delta, float* out, int size) {
    for (int i = 0; i < size; i++) {
        const Qint8 x_val = x[i];
        out[i] = x_val * delta;
    }
}

} // namespace ops

} // namespace gten