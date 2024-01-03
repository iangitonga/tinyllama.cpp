#pragma once


#include <cmath>
#include <memory>

#include "gten_types.h"
#include "log.h"

namespace gten {

namespace globs {
static const int q8_block_size = 64;
}

class Qparams {
public:
    Qparams() = default;

    Qparams(int n_rows, int n_cols) {
        GTEN_ASSERT(n_cols % globs::q8_block_size == 0);

        n_rows_ = n_rows;
        n_cols_ = n_cols;
        blocks_per_row_ = n_cols / globs::q8_block_size;
        const int n_deltas = n_rows_ * blocks_per_row_;
        deltas_ = std::shared_ptr<Float16[]>(new Float16[n_deltas]);
    }

    Float16* deltas() { return deltas_.get(); }
    const Float16* deltas() const { return deltas_.get(); }

    Float16* row_deltas(int row_idx) {
        GTEN_ASSERT(row_idx >= 0);
        GTEN_ASSERT(row_idx < n_rows_);

        const int deltas_per_row = blocks_per_row_;
        Float16* row_deltas_ptr = deltas_.get() + row_idx * deltas_per_row;
        return row_deltas_ptr;
    }

    const Float16* row_deltas(int row_idx) const {
        assert(row_idx >= 0);
        assert(row_idx <= n_rows_);
        
        const int deltas_per_row = blocks_per_row_;
        const Float16* row_deltas_ptr = deltas_.get() + row_idx * deltas_per_row;
        return row_deltas_ptr;
    }

    int nbytes() const { return n_rows_ * blocks_per_row_ * sizeof(Float16); }

    int n_deltas() const { return n_rows_ * blocks_per_row_; }
    int blocks_per_row() const { return blocks_per_row_; }

private:
    int n_rows_ = 0;
    int n_cols_ = 0;
    int blocks_per_row_ = 0;
    std::shared_ptr<Float16[]> deltas_;
};



namespace ops {

[[nodiscard]]
inline float quantize_row(const float* x, Qint8* out, int block_size) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < block_size; j++) {
        const float v = x[j];
        amax = std::max(amax, fabsf(v));
    }

    const float delta = amax / 127.0f;
    const float id = delta ? 1.0f/delta : 0.0f;

    for (int j = 0; j < block_size; ++j) {
        const float x0 = x[j] * id;
        out[j] = std::roundf(x0);
    }

    return delta;
}


[[nodiscard]]
inline Qint8 quantize(float x, float delta) {
    const float id = delta ? 1.0f/delta : 0.0f;

    const float x0 = x * id;
    const Qint8 quantized = roundf(x0);

    return quantized;
}

[[nodiscard]]
inline float compute_quantization_delta(const float* x, int size) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < size; j++) {
        const float v = x[j];
        amax = std::max(amax, fabsf(v));
    }

    const float delta = amax / 127.0f;
    return delta;
}


/// @brief Dequantize input quants of size `block_size` to floats.
inline void dequantize_block(const Qint8* x, float delta, float* out, int block_size) {
    for (int j = 0; j < block_size; ++j) {
        out[j] = x[j] * delta;
    }
}


[[nodiscard]]
inline float dequantize(Qint8 x, float delta) {
    return x * delta;
}


inline void quantize_row(int row_idx, const float* x, Qparams& out_qparams, Qint8* out) {
    const int blocks_per_row = out_qparams.blocks_per_row();
    const int block_size = globs::q8_block_size;

    Float16* out_row_deltas = out_qparams.row_deltas(row_idx);

    for (int i = 0; i < blocks_per_row; i++) {
        const float* x_block = x + i * block_size;

        float absmax = 0;
        for (int j = 0; j < block_size; j++) {
            const float x_val = x_block[j];
            absmax = std::max(absmax, fabsf(x_val));
        }

        const float delta = absmax / 127.0f;
        const float scale = delta ? 1.0f/delta : 0.0f;
         
        for (int j = 0; j < block_size; ++j) {
            const float x_val = x_block[j] * scale;
            out[i * block_size + j] = static_cast<Qint8>(roundf(x_val));
        }

        out_row_deltas[i] = fp32_to_fp16(delta);
    }
}

// Quantize a contigous row of floats and store it as a column in `out`.
inline void quantize_row_as_col(int row_idx, const float* x, Qparams& out_qparams, Qint8* out, const int stride) {
    const int blocks_per_row = out_qparams.blocks_per_row();
    const int block_size = globs::q8_block_size;

    Float16* out_row_deltas = out_qparams.row_deltas(row_idx);

    for (int i = 0; i < blocks_per_row; i++) {
        const float* x_block = x + i * block_size;

        float absmax = 0;
        for (int j = 0; j < block_size; j++) {
            const float x_val = x_block[j];
            absmax = std::max(absmax, fabsf(x_val));
        }

        const float delta = absmax / 127.0f;
        const float scale = delta ? 1.0f/delta : 0.0f;
         
        for (int j = 0; j < block_size; ++j) {
            const float x_val = x_block[j] * scale;
            const int out_idx = (i * block_size + j) * stride + row_idx;
            out[out_idx] = static_cast<Qint8>(roundf(x_val));
        }

        out_row_deltas[i] = fp32_to_fp16(delta);
    }
}

inline void dequantize_row(int row_idx, const Qint8* x, const Qparams& x_qparams, float* out) {
    const int blocks_per_row = x_qparams.blocks_per_row();
    const int block_size = globs::q8_block_size;

    const Float16* out_row_deltas = x_qparams.row_deltas(row_idx);

    for (int i = 0; i < blocks_per_row; i++) {
        const Qint8* x_block = x + i * block_size;
        const float delta = fp16_to_fp32(out_row_deltas[i]);

        for (int j = 0; j < block_size; j++) {
            out[i * block_size + j] = x_block[j] * delta;
        }
    }
}

// Dequantizes a contiguos column x and stores the result in out.
inline void dequantize_col(int col_idx, const Qint8* x, const Qparams& x_qparams, float* out, int size) {
    const int blocks_per_row = x_qparams.blocks_per_row();
    const int block_size = globs::q8_block_size;

    const Float16* deltas = x_qparams.deltas();

    for (int i = 0; i < size; i++) {
        const int delta_row = i * blocks_per_row;
        const int delta_block = col_idx / block_size;
        const float delta = fp16_to_fp32(deltas[delta_row + delta_block]);

        out[i] = x[i] * delta;
    }
}

inline void dequantize_row_scale(const Qint8* x, float delta, float* out, int size) {
    for (int i = 0; i < size; i++) {
        const Qint8 x_val = x[i];
        out[i] = x_val * delta;
    }
}

} // namespace ops

} // namespace gten