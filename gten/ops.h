#include <cstring>

#include "log.h"
#include "quants.h"
#include "tensor.h"
#include "simd_ops.h"

#ifdef _OPENMP
#define GTEN_OMP 1
#include <omp.h>
#endif

namespace gten {
namespace ops {


// Stores buffers required by ops.
class OpsState {
public:
    OpsState() {
        // max bufsize op: gelu 2 x n_embd * 4 == 2 * n_mlp
        const int max_bufsize = 1024 * 1024;
        buf_ = new float[max_bufsize];
        buf_numel_ = max_bufsize;
    }
    ~OpsState() { delete[] buf_; }

    // Obtain a ptr to a buffer of size `numel` * sizeof(float).
    float* buf(int numel) const {
        GTEN_ASSERT(numel <= buf_numel_);
        return buf_;
    }

private:
    float* buf_ = nullptr;
    int buf_numel_ = 0;
};

static const OpsState g_ops_state = OpsState();


static void vec_add_f16(const Float16* a, const Float16* b, Float16* out, int vec_size)
{
#ifdef GTEN_SIMD_AVX
    const int simd_vec_size = (vec_size / GTEN_SIMD_VEC_SIZE) * GTEN_SIMD_VEC_SIZE;

    for (int i = 0; i < simd_vec_size; i += GTEN_SIMD_VEC_SIZE) {
        Vec_f32x8 x0 = vec_f32x8_load(a + i);
        Vec_f32x8 x1 = vec_f32x8_load(b + i);
        Vec_f32x8 x_sum = vec_f32x8_add(x0, x1);
        vec_f32x8_store(x_sum, out + i);
    }

    for (int i = simd_vec_size; i < vec_size; i++) {
        const float x0 = fp16_to_fp32(a[i]);
        const float x1 = fp16_to_fp32(b[i]);
        out[i] = fp32_to_fp16(x0 + x1);
    }
#else

    for (int i = 0; i < vec_size; i++) {
        const float x0 = fp16_to_fp32(a[i]);
        const float x1 = fp16_to_fp32(b[i]);
        out[i] = fp32_to_fp16(x0 + x1);
    }

#endif
}


static void vec_add_f32(const float* a, const float* b, float* out, int vec_size)
{
#ifdef GTEN_SIMD_AVX

    const int simd_vec_size = (vec_size / GTEN_SIMD_VEC_SIZE) * GTEN_SIMD_VEC_SIZE;

    for (int i = 0; i < simd_vec_size; i += GTEN_SIMD_VEC_SIZE) {
        Vec_f32x8 x0 = vec_f32x8_load(a + i);
        Vec_f32x8 x1 = vec_f32x8_load(b + i);
        Vec_f32x8 x_sum = vec_f32x8_add(x0, x1);
        vec_f32x8_store(x_sum, out + i);
    }

    for (int i = simd_vec_size; i < vec_size; i++) {
        const float x0 = a[i];
        const float x1 = b[i];
        out[i] = x0 + x1;
    }

#else
    const int unrolled_vec_size = (vec_size / 8) * 8;

    for (int i = 0; i < unrolled_vec_size; i += 8) {
        out[i] = a[i] + b[i];
        out[i + 1] = a[i + 1] + b[i + 1];
        out[i + 2] = a[i + 2] + b[i + 2];
        out[i + 3] = a[i + 3] + b[i + 3];
        out[i + 4] = a[i + 4] + b[i + 4];
        out[i + 5] = a[i + 5] + b[i + 5];
        out[i + 6] = a[i + 6] + b[i + 6];
        out[i + 7] = a[i + 7] + b[i + 7];
    } 

    // leftovers
    for (int i = unrolled_vec_size; i < vec_size; i++) {
        out[i] = a[i] + b[i];
    }

#endif
}

static float vec_dot_product_f16(const Float16* vec_a, const Float16* vec_b, int vec_size)
{
#ifdef GTEN_SIMD_AVX

    const int simd_vec_size = (vec_size / GTEN_SIMD_VEC_SIZE) * GTEN_SIMD_VEC_SIZE;
    
    Vec_f32x8 dot_prod_accum = vec_f32x8_setzero();
    for (int i = 0; i < simd_vec_size; i += GTEN_SIMD_VEC_SIZE) {
        Vec_f32x8 x0 = vec_f32x8_load(vec_a + i);
        Vec_f32x8 x1 = vec_f32x8_load(vec_b + i);
        // dot_prod += vec_f32x8_sum(vec_f32x8_mul(x0, x1));
        dot_prod_accum = vec_f32x8_fma(x0, x1, dot_prod_accum);
    }
    
    float dot_prod = vec_f32x8_sum(dot_prod_accum);

    for (int i = simd_vec_size; i < vec_size; i++) {
        const float x0 = fp16_to_fp32(vec_a[i]);
        const float x1 = fp16_to_fp32(vec_b[i]);
        dot_prod += x0 * x1;
    }

#else

    float dot_prod = 0.0f;

    for (int i = 0; i < vec_size; i += 1)
    {
        dot_prod += fp16_to_fp32(vec_a[i]) * fp16_to_fp32(vec_b[i]);
    }

#endif

    return dot_prod;
}


static float vec_dot_product_f32(const float* vec_a, const float* vec_b, int vec_size)
{
#ifdef GTEN_SIMD_AVX

    const int simd_vec_size = (vec_size / GTEN_SIMD_VEC_SIZE) * GTEN_SIMD_VEC_SIZE;
    
    Vec_f32x8 dot_prod_accum = vec_f32x8_setzero();
    for (int i = 0; i < simd_vec_size; i += GTEN_SIMD_VEC_SIZE) {
        Vec_f32x8 x0 = vec_f32x8_load(vec_a + i);
        Vec_f32x8 x1 = vec_f32x8_load(vec_b + i);
        // dot_prod += vec_f32x8_sum(vec_f32x8_mul(x0, x1));
        dot_prod_accum = vec_f32x8_fma(x0, x1, dot_prod_accum);
    }
    
    float dot_prod = vec_f32x8_sum(dot_prod_accum);

    for (int i = simd_vec_size; i < vec_size; i++) {
        const float x0 = vec_a[i];
        const float x1 = vec_b[i];
        dot_prod += x0 * x1;
    }

# else
    const int unrolled_vec_size = (vec_size / 8) * 8;

    float dot_prod = 0.0f;
    for (int i = 0; i < unrolled_vec_size; i += 8) {
        dot_prod += vec_a[i] * vec_b[i];
        dot_prod += vec_a[i + 1] * vec_b[i + 1];
        dot_prod += vec_a[i + 2] * vec_b[i + 2];
        dot_prod += vec_a[i + 3] * vec_b[i + 3];
        dot_prod += vec_a[i + 4] * vec_b[i + 4];
        dot_prod += vec_a[i + 5] * vec_b[i + 5];
        dot_prod += vec_a[i + 6] * vec_b[i + 6];
        dot_prod += vec_a[i + 7] * vec_b[i + 7];
    }

    // leftovers
    for (int i = unrolled_vec_size; i < vec_size; i++) {
        dot_prod += vec_a[i] * vec_b[i];
    }

#endif
    return dot_prod;
}


static float vec_dot_product_q8(const Qint8* a, const Float16* a_ds, const Qint8* b, const Float16* b_ds, int blk_size, int vec_size)
{
    // GTEN_ASSERTM(vec_size % blk_size == 0, "row size: %d is incompatible with block size: %d", vec_size, blk_size);

    // Ensure that when the input vector is inside a block rather than the
    // vector having multiple blocks inside it, we still make the right computations.
    const int nblocks = blk_size < vec_size ? vec_size / blk_size : 1;
    blk_size = blk_size < vec_size ? blk_size : vec_size;

    // blk_size % 8 == 0 || blk_sizw % 16 == 0

#ifdef GTEN_SIMD_AVX
    // Dot product accumulator with 4 slots. The sum of the four accumulators gives the
    // total dot product.
    __m128 dot_accum = _mm_set1_ps(0.0f);

    for (int i = 0; i < nblocks; i++)
    {
        // dotprod = aq0*ad * bq0*bd + aq1*ad * bq1*bd + ... + aqN*ad + bqN*bd
        //         = adbd(aq0 * bq0) + adbd(aq1 * bq1) + ... + adbd(aqN * bqN)
        //         = adbd(aq0 * bq0 + aq1 * bq1 + ... + aqN * bqN)
        // We compute integer arithmetic inside the brackets and scaled by the block
        // quantisation deltas.

        // Integer dot product accumulator for current block.
        __m128i blk_dot_accum = _mm_set1_epi32(0);

        for (int j = 0; j < blk_size; j += 16)
        {
            const int idx_offs = i * blk_size + j;

            // Load 64-bit(8 1-byte quants) in the lower half. [8-quants, -------].
            const __m128i a00 = _mm_loadu_si64(a + idx_offs);
            const __m128i a01 = _mm_loadu_si64(a + idx_offs + 8);

            const __m128i b00 = _mm_loadu_si64(b + idx_offs);
            const __m128i b01 = _mm_loadu_si64(b + idx_offs + 8);

            // Convert 8 quants in the lower half to 16-bit ints.
            const __m128i a02 = _mm_cvtepi8_epi16(a00);
            const __m128i a03 = _mm_cvtepi8_epi16(a01);

            const __m128i b02 = _mm_cvtepi8_epi16(b00);
            const __m128i b03 = _mm_cvtepi8_epi16(b01);

            // Multiply the 8 16-bit ints to obtain 8 32-bit ints and add adjacent
            // values to obtain 4 32-bit ints.
            // TODO: Can we instead do 16-bit to 16-bit e.g _mullo_epi16
            const __m128i c00 = _mm_madd_epi16(a02, b02);
            const __m128i c01 = _mm_madd_epi16(a03, b03);

            // Add the results and add the output to the accumulator.
            const __m128i c02 = _mm_add_epi32(c00, c01);
            blk_dot_accum = _mm_add_epi32(blk_dot_accum, c02);
        }

        // const __m128 a_blk_delta = _mm_broadcast_ss(a_ds + i);
        // const __m128 b_blk_delta = _mm_broadcast_ss(b_ds + i); 
        // const __m128 ab_blk_delta = _mm_mul_ps(a_blk_delta, b_blk_delta);
        const __m128 ab_blk_delta = _mm_set1_ps(fp16_to_fp32(a_ds[i]) * fp16_to_fp32(b_ds[i]));

        const __m128 blk_dot_accum_f = _mm_cvtepi32_ps(blk_dot_accum);
        dot_accum = _mm_add_ps(dot_accum, _mm_mul_ps(blk_dot_accum_f, ab_blk_delta));
    }

    const __m128 dotsum0 = _mm_hadd_ps(dot_accum, dot_accum);
    const __m128 dotsum1 = _mm_hadd_ps(dotsum0, dotsum0);
    const float dot_prod = _mm_cvtss_f32(dotsum1);

#else

    float dot_prod = 0.0f;

    for (int i = 0; i < nblocks; i++)
    {
        // accumulator for integer block-dot products.
        int blk_dot_prod_i[2] = {0, 0};

        for (int j = 0; j < blk_size; j += 8) {
            const int idx = i * blk_size + j;
            blk_dot_prod_i[0] += a[idx] * b[idx];
            blk_dot_prod_i[1] += a[idx + 1] * b[idx + 1];
            blk_dot_prod_i[0] += a[idx + 2] * b[idx + 2];
            blk_dot_prod_i[1] += a[idx + 3] * b[idx + 3];
            blk_dot_prod_i[0] += a[idx + 4] * b[idx + 4];
            blk_dot_prod_i[1] += a[idx + 5] * b[idx + 5];
            blk_dot_prod_i[0] += a[idx + 6] * b[idx + 6];
            blk_dot_prod_i[1] += a[idx + 7] * b[idx + 7];
        }

        const float blk_dot_prod = float(blk_dot_prod_i[0] + blk_dot_prod_i[1]);
        const float a_delta = fp16_to_fp32(a_ds[i]);
        const float b_delta = fp16_to_fp32(b_ds[i]);
        dot_prod += blk_dot_prod * a_delta * b_delta;
    }
#endif

    return dot_prod;
}

static void tensor_row_index_impl_f16(const Tensor& src, const Tensor& indices, Tensor& out, int last_token_only)
{
    const Float16* w_data = src.data_ptr<Float16>();
    const int* indices_data = indices.data_ptr<int>();
    Float16* out_data = out.data_ptr<Float16>();
    const int rowsize = src.size(1);
    const size_t rowsizebytes = rowsize * src.itemsize();

    const int n_ctx = indices.size(0);
    const int ctx_start = last_token_only ? n_ctx - 1 : 0;
    for (int i = ctx_start; i < indices.numel(); i++) {
        const void* src_row_data = w_data + indices_data[i] * rowsize;
        void* out_row_data = out_data + i * rowsize;
        std::memcpy(out_row_data, src_row_data, rowsizebytes);
    }
}

static void tensor_row_index_impl_q8(const Tensor& src, const Tensor& indices, Tensor& out, bool last_token_only)
{
    const Qint8* src_data = src.data_ptr<Qint8>();
    const int* indices_data = indices.data_ptr<int>();
    Qint8* out_data = out.data_ptr<Qint8>();

    const int rowsize = src.size(1);
    const size_t rowsizebytes = rowsize * src.itemsize();

    const Qparams& w_qparams = src.qparams();
    Qparams& out_qparams = out.qparams();

    const int block_size = w_qparams.block_size();
    const int n_blocks = rowsize / block_size;

    const int n_ctx = indices.size(0);
    const int ctx_start = last_token_only ? n_ctx - 1 : 0;
    for (int i = ctx_start; i < indices.numel(); i++) {
        const Qint8* src_row_data = src_data + indices_data[i] * rowsize;
        Qint8* dest_row_data = out_data + i * rowsize;

        std::memcpy(dest_row_data, src_row_data, rowsizebytes);

        // Copy row quantization deltas.
        const Float16* w_row_deltas = w_qparams.row_deltas(indices_data[i]);
        Float16* out_row_deltas = out_qparams.row_deltas(i);
        std::memcpy(out_row_deltas, w_row_deltas, n_blocks * sizeof(Float16));
    }
}

/// @brief Copies the indexed rows of the source tensor to output tensor.
/// @param src A 2-d tensor to be indexed.
/// @param indices A 1-d tensor of indices with dtype = int.
/// @param out A 2d tensor with enough capacity to fit the indexed rows. Its dtype
///  must be the same as source tensor.
/// @param last_token_only Whether to index the last token only, if others are cached.
void token_embed(const Tensor& weight, const Tensor& tokens, Tensor& out, bool last_token_only = false)
{
    GTEN_ASSERT(weight.is_2d());
    GTEN_ASSERT(tokens.is_1d() && tokens.dtype() == kInt32);
    const int n_ctx = tokens.size(0);
    const int n_embd = weight.size(1);
    GTEN_ASSERT(out.shape_eq({n_ctx, n_embd}));
    GTEN_ASSERT(weight.dtype() == out.dtype());

    if (weight.is_quantized()) {
        tensor_row_index_impl_q8(weight, tokens, out, last_token_only);
    } else {
        tensor_row_index_impl_f16(weight, tokens, out, last_token_only);
    }
}

static void emb_matmul_impl_q8(const Tensor& x, const Tensor& w, Tensor& out)
{
    const Qint8* x_data = x.data_ptr<Qint8>();
    const Qint8* w_data = w.data_ptr<Qint8>();
    float* out_data = out.data_ptr<float>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int n_vocab = w.size(0);

    const Qparams& x_qparams = x.qparams();
    const Qparams& w_qparams = w.qparams();

    const int block_size = x_qparams.block_size();

#ifdef GTEN_OMP
    #pragma omp parallel for collapse(2)
#endif
    for (int xrow = n_ctx-1; xrow < n_ctx; xrow++) {
        for (int wrow = 0; wrow < n_vocab; wrow++) {
            const Qint8* x_row_data = x_data + xrow * n_embd;
            const Float16* x_row_deltas = x_qparams.row_deltas(xrow);
            const Qint8* w_row_data = w_data + wrow * n_embd;
            const Float16* w_row_deltas = w_qparams.row_deltas(wrow);

            const float dot_prod = vec_dot_product_q8(x_row_data, x_row_deltas, w_row_data, w_row_deltas, block_size, n_embd);
            out_data[wrow] = dot_prod;
        }
    }
}

static void emb_matmul_impl_f16(const Tensor& x, const Tensor& w, Tensor& out)
{
    const Float16* x_data = x.data_ptr<Float16>();
    const Float16* w_data = w.data_ptr<Float16>();
    float* out_data = out.data_ptr<float>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int n_vocab = w.size(0);

#ifdef GTEN_OMP
    #pragma omp parallel for collapse(2)
#endif
    for (int xrow = n_ctx-1; xrow < n_ctx; xrow++) {
        for (int wrow = 0; wrow < n_vocab; wrow++) {
            const Float16* x_row_data = x_data + xrow * n_embd;
            const Float16* w_row_data = w_data + wrow * n_embd;
            float dot_prod = vec_dot_product_f16(x_row_data, w_row_data, n_embd);
            out_data[wrow] = dot_prod;
        }
    }
}

/// @brief Computes a matmul between input's last ctx vector and emb table to produce logits
///   for the next token.
/// @param x Input tensor of shape (n_ctx, n_embd).
/// @param w Embedding table tensor of shape (n_vocab, n_embd).
/// @param out Output tensor of shape (n_vocab).
static void emb_matmul(const Tensor& x, const Tensor& weight, Tensor& out) {
    GTEN_ASSERT(x.is_2d());
    const int n_embd = x.size(1);
    GTEN_ASSERT(weight.is_2d() && weight.size(1) == n_embd);
    const int n_vocab = weight.size(0);
    GTEN_ASSERT(out.is_1d() && out.dtype() == kFloat32 && out.size(0) == n_vocab);

    if (weight.is_quantized()) {
        emb_matmul_impl_q8(x, weight, out);
    } else {
        emb_matmul_impl_f16(x, weight, out);
    }
}

// matmul
// add

static void matmul_2d_impl_q8(const Tensor& x, const Tensor& w, Tensor& out, const bool last_ctx_only)
{
    const Qint8* x_data = x.data_ptr<Qint8>();
    const Qint8* w_data = w.data_ptr<Qint8>();
    Qint8* out_data = out.data_ptr<Qint8>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int d_out = w.size(0);
    const int x_st0 = x.stride(0);
    const int w_st0 = w.stride(0);
    const int out_st0 = out.stride(0);

    const Qparams& x_qparams = x.qparams();
    const Qparams& w_qparams = w.qparams();
    Qparams& out_qparams = out.qparams();

    const int block_size = x_qparams.block_size();

    float* out_buf = g_ops_state.buf(d_out);

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    for (int xrow = ctx_start; xrow < n_ctx; xrow++) {
        for (int wrow = 0; wrow < d_out; wrow++) {
            const Qint8* xrow_data = x_data + xrow * x_st0;
            const Float16* x_row_deltas = x_qparams.row_deltas(xrow);
            const Qint8* wrow_data = w_data + wrow * w_st0;
            const Float16* w_row_deltas = w_qparams.row_deltas(wrow);

            const float dot_prod = vec_dot_product_q8(xrow_data, x_row_deltas, wrow_data, w_row_deltas, block_size, n_embd);

            out_buf[wrow] = dot_prod;
        }

        Qint8* outrow_data = out_data + xrow * out_st0;
        quantize_row(xrow, out_buf, out_qparams, outrow_data);
    }
}

static void matmul_2d_impl_f16(const Tensor& x, const Tensor& w, Tensor& out, const bool last_ctx)
{
    const Float16* x_data = x.data_ptr<Float16>();
    const Float16* w_data = w.data_ptr<Float16>();
    Float16* out_data = out.data_ptr<Float16>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int d_out = w.size(0);
    const int x_st0 = x.stride(0);
    const int w_st0 = w.stride(0);
    const int out_st0 = out.stride(0);

    const int ctx_start = last_ctx ? n_ctx - 1 : 0;

#ifdef GTEN_OMP
    #pragma omp parallel for collapse(2)
#endif
    for (int xrow = ctx_start; xrow < n_ctx; xrow++) {
        for (int wrow = 0; wrow < d_out; wrow++) {
            const Float16* xrow_data = x_data + xrow * x_st0;
            const Float16* wrow_data = w_data + wrow * w_st0;
            float dot_prod = vec_dot_product_f16(xrow_data, wrow_data, n_embd);
            out_data[xrow * out_st0 + wrow] = fp32_to_fp16(dot_prod);
        }
    }
}


static void matmul_2d(const Tensor& x, const Tensor& w, Tensor& out, const bool last_ctx = false)
{
    const int n_ctx = x.size(0);
    const int n_out = w.size(0);
    const int n_embd = x.size(1);

    GTEN_ASSERT(x.is_2d());
    GTEN_ASSERT(w.is_2d() && w.size(1) == n_embd);
    GTEN_ASSERT(out.is_2d() && out.shape_eq({n_ctx, n_out}));
    GTEN_ASSERT(x.dtype() == w.dtype() && w.dtype() == out.dtype());

    if (x.is_quantized()) {
        matmul_2d_impl_q8(x, w, out, last_ctx);
    } else {
        matmul_2d_impl_f16(x, w, out, last_ctx);
    }
}

void matmul_2d_transposed_q8(const Tensor& x, const Tensor& w, Tensor& out, const bool last_ctx_only)
{
    const Qint8* x_data = x.data_ptr<Qint8>();
    const Qint8* w_data = w.data_ptr<Qint8>();
    Qint8* out_data = out.data_ptr<Qint8>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int d_out = w.size(0);
    const int x_st0 = x.stride(0);
    const int w_st0 = w.stride(0);
    const int out_st0 = out.stride(0);  // out: nemb, nctx: max_ctx, 1

    const Qparams& x_qparams = x.qparams();
    const Qparams& w_qparams = w.qparams();

    const int block_size = out.qparams().block_size();
    const int n_blocks = d_out / block_size;

    float* out_buf = g_ops_state.buf(d_out);

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    
    for (int xrow = ctx_start; xrow < n_ctx; xrow++) {
        const Qint8* x_row_data = x_data + xrow * x_st0;
        const Float16* x_ds = x_qparams.row_deltas(xrow);

        for (int wrow = 0; wrow < d_out; wrow++) {
            const Qint8* w_row_data = w_data + wrow * w_st0;
            const Float16* w_ds = w_qparams.row_deltas(wrow);

            const float dot_prod = vec_dot_product_q8(x_row_data, x_ds, w_row_data, w_ds, block_size, n_embd);
            
            out_buf[wrow] = dot_prod;
        }

        Float16* out_row_deltas = out.qparams().row_deltas(xrow);

        // Quantize the row but store it as a column (i.e transposed). 
        for (int i = 0; i < n_blocks; i++) {
            const float delta = compute_quantization_delta(out_buf + i * block_size, block_size);

            out_row_deltas[i] = fp32_to_fp16(delta);

            for (int j = 0; j < block_size; j++) {
                const int wrow = i * block_size + j;
                out_data[wrow * out_st0 + xrow] = quantize(out_buf[i * block_size + j], delta);
            }
        }
    }
}


void matmul_2d_transposed_f16(const Tensor& x, const Tensor& w, Tensor& out, const bool last_ctx=false)
{
    const Float16* x_data = x.data_ptr<Float16>();
    const Float16* w_data = w.data_ptr<Float16>();
    Float16* out_data = out.data_ptr<Float16>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int d_out = w.size(0);
    const int x_st0 = x.stride(0);
    const int w_st0 = w.stride(0);
    const int out_st0 = out.stride(0);

    const int ctx_start = last_ctx ? n_ctx - 1 : 0;

#ifdef GTEN_OMP
    #pragma omp parallel for collapse(2)
#endif
    for (int xrow = ctx_start; xrow < n_ctx; xrow++) {
        for (int wrow = 0; wrow < d_out; wrow++) {
            const Float16* x_row_data = x_data + xrow * x_st0;
            const Float16* w_row_data = w_data + wrow * w_st0;
            const float dot_prod = vec_dot_product_f16(x_row_data, w_row_data, n_embd);
            out_data[wrow * out_st0 + xrow] = fp32_to_fp16(dot_prod);
        }
    }
}

static void matmul_2d_transposed(const Tensor& x, const Tensor& w, Tensor& out, const bool last_ctx=false)
{
    const int n_ctx = x.size(0);
    const int n_out = w.size(0);
    const int n_embd = x.size(1);

    GTEN_ASSERT(x.is_2d());
    GTEN_ASSERT(w.is_2d() && w.size(1) == n_embd);
    GTEN_ASSERT(out.is_2d() && out.shape_eq({n_out, n_ctx}));
    GTEN_ASSERT(x.dtype() == w.dtype() && w.dtype() == out.dtype());

    if (x.is_quantized()) {
        matmul_2d_transposed_q8(x, w, out, last_ctx);
    } else {
        matmul_2d_transposed_f16(x, w, out, last_ctx);
    }
}


static void silu_q8(const Tensor& inp, Tensor& out)
{
    const Qint8* inp_data = inp.data_ptr<Qint8>();
    Qint8* out_data = out.data_ptr<Qint8>();

    const int n_ctx = inp.size(0);
    const int n_embd = inp.size(1);

    const Qparams& inp_qparams = inp.qparams();
    Qparams& out_qparams = out.qparams();

    float* inp_buf = g_ops_state.buf(n_embd * 2);
    float* out_buf = inp_buf + n_embd;

    for (int i = 0; i < n_ctx; i++)
    {
        const Qint8* inp_row_data = inp_data + i * n_embd;
        Qint8* out_row_data = out_data + i * n_embd;
        dequantize_row(i, inp_row_data, inp_qparams, inp_buf);

        for (int j = 0; j < n_embd; j++) {
            const float x = inp_buf[j];
            out_buf[j] = x / (1.0f + std::exp(-x));
        }

        quantize_row(i, out_buf, out_qparams, out_row_data);
    }
}


static void silu_f16(const Tensor& inp, Tensor& out) {
    const Float16* inp_data = inp.data_ptr<Float16>();
    Float16* out_data = out.data_ptr<Float16>();

    const int numel = inp.numel();

    for (int i = 0; i < numel; ++i)
    {
        // sigmoid
        const float x = fp16_to_fp32(inp_data[i]);
        // const float x_sigmoid = x < 0.0f ? std::exp(x) / (1.0f + std::exp(x)) : 1.0f / (1.0f + std::exp(-x));
        const float x_sigmoid =  x / (1.0f + std::exp(-x));
        out_data[i] = fp32_to_fp16(x_sigmoid);
    }
}


static void silu(const Tensor& inp, Tensor& out)
{
    GTEN_ASSERT(inp.shape_eq(out.shape()));
    GTEN_ASSERT(inp.dtype() == out.dtype());

    if (inp.is_quantized()) {
        silu_q8(inp, out);
    } else {
        silu_f16(inp, out);
    }
}


static void rotary_emb_q8(Tensor& inp, int start_pos=0)
{
    const int n_ctx = inp.size(0);
    const int n_embd = inp.size(1);
    const int d_head = 64;
    const int n_head = n_embd / d_head;

    Tensor inpx = inp.view({n_ctx, n_head, d_head}); // .permute({1, 0, 2})

    const int st0 = inpx.stride(0);

    Qint8* inp_data = inpx.data_ptr<Qint8>();

    float* inp_buf = g_ops_state.buf(n_embd);

    Qparams& inp_qparams = inp.qparams();

    const float d = static_cast<float>(d_head);

    for (int i = start_pos; i < n_ctx; ++i) {
        Qint8* inp_row_data = inp_data + i*st0;
        dequantize_row(i, inp_row_data, inp_qparams, inp_buf);

       for (int h = 0; h < n_head; ++h) {
            float* inp_vec = inp_buf + h*d_head;
            const float m = static_cast<float>(i);

            const int d_half = d_head / 2;
            for (int j = 0; j < d_half; ++j)
            {
                const float x0 = inp_vec[j];
                const float x1 = inp_vec[j + d_half];

                const float m_theta_i = m * std::pow(10000.0f, -(2.0f*j/d));

                const float o0 = x0 * std::cos(m_theta_i) - x1 * std::sin(m_theta_i);
                const float o1 = x0 * std::sin(m_theta_i) + x1 * std::cos(m_theta_i);

                inp_vec[j] = o0;
                inp_vec[j + d_half] = o1;
            }
        }

        quantize_row(i, inp_buf, inp_qparams, inp_row_data);
    }
}


static void rotary_emb_f16(Tensor& inp, int start_pos=0)
{
    const int n_ctx = inp.size(0);
    const int n_embd = inp.size(1);
    const int d_head = 64;
    const int n_head = n_embd / d_head;

    Tensor inpx = inp.view({n_ctx, n_head, d_head}).permute({1, 0, 2});

    const int st0 = inpx.stride(0);
    const int st1 = inpx.stride(1);

    Float16* inp_data = inpx.data_ptr<Float16>();

    const float d = static_cast<float>(d_head);

    for (int h = 0; h < n_head; ++h) {
        for (int i = start_pos; i < n_ctx; ++i)
        {
            Float16* inp_vec = inp_data + h*st0 + i*st1;
            const float m = static_cast<float>(i);

            const int d_half = d_head / 2;
            for (int j = 0; j < d_half; ++j)
            {
                const float x0 = fp16_to_fp32(inp_vec[j]);
                const float x1 = fp16_to_fp32(inp_vec[j + d_half]);

                const float m_theta_i = m * std::pow(10000.0f, -(2.0f*j/d));

                const float o0 = x0 * std::cos(m_theta_i) - x1 * std::sin(m_theta_i);
                const float o1 = x0 * std::sin(m_theta_i) + x1 * std::cos(m_theta_i);

                inp_vec[j] = fp32_to_fp16(o0);
                inp_vec[j + d_half] = fp32_to_fp16(o1);
            }
        }
    }
}

static void rotary_emb(Tensor& inp, int start_pos=0)
{
    if (inp.is_quantized()) {
        rotary_emb_q8(inp, start_pos);
    } else {
        rotary_emb_f16(inp, start_pos);
    }
}

static void rms_norm_vec_f32(const float* inp, const Float16* weight, float* out, const int vec_size) {
    float sq_sum = 0.0f;

    for (int i = 0; i < vec_size; ++i) {
        sq_sum += std::pow(inp[i], 2.0f);
    }

    const float sq_mean = sq_sum / static_cast<float>(vec_size);
    const float root_mean_sq = std::sqrt(sq_mean);

    for (int i = 0; i < vec_size; ++i)
    {
        const float xi = inp[i];
        const float wi = fp16_to_fp32(weight[i]);
        out[i] = xi / (root_mean_sq + 1e-6f) * wi;
    }
}

static void rms_norm_vec_f16(const Float16* inp, const Float16* weight, Float16* out, const int vec_size) {
    float sq_sum = 0.0f;

    for (int i = 0; i < vec_size; ++i) {
        sq_sum += std::pow(fp16_to_fp32(inp[i]), 2.0f);
    }

    const float sq_mean = sq_sum / static_cast<float>(vec_size);
    const float root_mean_sq = std::sqrt(sq_mean);

    for (int i = 0; i < vec_size; ++i)
    {
        const float xi = fp16_to_fp32(inp[i]);
        const float wi = fp16_to_fp32(weight[i]);
        out[i] = fp32_to_fp16(xi / (root_mean_sq + 1e-6f) * wi);
    }
}


static void rms_norm_q8(const Tensor& inp, const Tensor& weight, Tensor& out, int start_pos)
{
    const Qint8* inp_data = inp.data_ptr<Qint8>();
    const Float16* weight_data = weight.data_ptr<Float16>();
    Qint8* out_data = out.data_ptr<Qint8>();

    const int n_embd = inp.size(1);
    float* inp_buf = g_ops_state.buf(n_embd + n_embd);
    float* out_buf = inp_buf + n_embd;

    const Qparams& inp_qparams = inp.qparams();
    Qparams& out_qparams = out.qparams();

    const int n_ctx = inp.size(0);
    for (int i = start_pos; i < n_ctx; ++i)
    {
        const Qint8* inp_row_data = inp_data + i * n_embd;
        dequantize_row(i, inp_row_data, inp_qparams, inp_buf);

        rms_norm_vec_f32(inp_buf, weight_data, out_buf, n_embd);

        Qint8* out_row_data = out_data + i * n_embd;
        quantize_row(i, out_buf, out_qparams, out_row_data);
    }
}

static void rms_norm_f16(const Tensor& inp, const Tensor& weight, Tensor& out, int start_pos)
{
    const Float16* inp_data = inp.data_ptr<Float16>();
    const Float16* weight_data = weight.data_ptr<Float16>();
    Float16* out_data = out.data_ptr<Float16>();

    const int n_ctx = inp.size(0);
    const int n_embd = inp.size(1);

    for (int i = start_pos; i < n_ctx; ++i)
    {
        const Float16* inp_row_data = inp_data + i * n_embd;
        Float16* out_row_data = out_data + i * n_embd;
        rms_norm_vec_f16(inp_row_data, weight_data, out_row_data, n_embd);
    }
}

static void rms_norm(const Tensor& inp, const Tensor& weight, Tensor& out, int start_pos = 0) {
    const int n_embd = inp.size(1);
    GTEN_ASSERT(weight.size(0) == n_embd);
    GTEN_ASSERT(inp.is_2d() && inp.dtype() == out.dtype());
    GTEN_ASSERT(weight.is_1d());
    GTEN_ASSERT(inp.shape_eq(out.shape()));

    if (inp.is_quantized()) {
        rms_norm_q8(inp, weight, out, start_pos);
    } else {
        rms_norm_f16(inp, weight, out, start_pos);
    }
}


static void vec_mul_f32(const float* a, const float* b, float* out, int vec_size) {
    for (int i = 0; i < vec_size; i++) {
        out[i] = a[i] * b[i];
    }
}


static void mul_q8(const Tensor& inp0, const Tensor& inp1, Tensor& out)
{
    const Qint8* inp0_data = inp0.data_ptr<Qint8>();
    const Qint8* inp1_data = inp1.data_ptr<Qint8>();
    Qint8* out_data = out.data_ptr<Qint8>();

    const int n_ctx = inp0.size(0);
    const int n_embd = inp0.size(1);

    const Qparams& inp0_qparams = inp0.qparams();
    const Qparams& inp1_qparams = inp1.qparams();
    Qparams& out_qparams = out.qparams();

    float* inp0_buf = g_ops_state.buf(n_embd * 3);
    float* inp1_buf = inp0_buf + n_embd;
    float* out_buf = inp1_buf + n_embd;

    for (int i = 0; i < n_ctx; i++) {
        const Qint8* inp0_row_data = inp0_data + i * n_embd;
        const Qint8* inp1_row_data = inp1_data + i * n_embd;

        dequantize_row(i, inp0_row_data, inp0_qparams, inp0_buf);
        dequantize_row(i, inp1_row_data, inp1_qparams, inp1_buf);

        vec_mul_f32(inp0_buf, inp1_buf, out_buf, n_embd);

        Qint8* out_row_data = out_data + i * n_embd;
        quantize_row(i, out_buf, out_qparams, out_row_data);
    }
}


static void mul_f16(const Tensor& inp0, const Tensor& inp1, Tensor& out)
{
    const Float16* inp0_data = inp0.data_ptr<Float16>();
    const Float16* inp1_data = inp1.data_ptr<Float16>();
    Float16* out_data = out.data_ptr<Float16>();

    const int numel = inp0.numel();

    for (int i = 0; i < numel; i++) {
        const float x0 = fp16_to_fp32(inp0_data[i]);
        const float x1 = fp16_to_fp32(inp1_data[i]);
        out_data[i] = fp32_to_fp16(x0 * x1);
    }
}


static void mul(const Tensor& inp0, const Tensor& inp1, Tensor& out)
{
    GTEN_ASSERT(inp0.dtype() == inp1.dtype() && inp1.dtype() == out.dtype());
    GTEN_ASSERT(inp0.shape_eq(inp1.shape()) && inp1.shape_eq(out.shape()));

    if (inp0.is_quantized()) {
        mul_q8(inp0, inp1, out);
    } else {
        mul_f16(inp0, inp1, out);
    }
}


static void add_impl_q8(const Tensor& x0, const Tensor& x1, Tensor& out, const bool last_ctx_only)
{
    const Qint8* x0_data = x0.data_ptr<Qint8>();
    const Qint8* x1_data = x1.data_ptr<Qint8>();
    Qint8* out_data = out.data_ptr<Qint8>();

    const int n_ctx = x0.size(0);
    const int n_embd = x0.size(1);
    const int st0 = x0.stride(0);

    Qparams& out_qparams = out.qparams();

    float* buf = g_ops_state.buf(n_embd * 3);
    float* x0_buf = buf;
    float* x1_buf = buf + n_embd;
    float* out_buf = buf + n_embd + n_embd;

    // auto [x0_buf, x1_buf, x2_buf] = g_ops_state.buf(n_embd, n_embd, n_embd)
    // Fewer lines of code. Just as efficient. no ptr arithmetic.

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    for (int i = ctx_start; i < n_ctx; i++)
    {
        const Qint8* x0_row_data = x0_data + i * st0;
        const Qint8* x1_row_data = x1_data + i * st0;

        dequantize_row(i, x0_row_data, x0.qparams(), x0_buf);
        dequantize_row(i, x1_row_data, x1.qparams(), x1_buf);

        vec_add_f32(x0_buf, x1_buf, out_buf, n_embd);

        Qint8* out_row_data = out_data + i * st0;
        quantize_row(i, out_buf, out_qparams, out_row_data);
    }
}

static void add_impl_f16(const Tensor& x0, const Tensor& x1, Tensor& out, const bool last_ctx_only)
{
    const Float16* x0_data = x0.data_ptr<Float16>();
    const Float16* x1_data = x1.data_ptr<Float16>();
    Float16* out_data = out.data_ptr<Float16>();

    const int n_ctx = x0.size(0);
    const int n_embd = x0.size(1);
    const int st0 = x0.stride(0);

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    const Float16* x0_offs_data = x0_data + ctx_start * st0;
    const Float16* x1_offs_data = x1_data + ctx_start * st0;
    Float16* out_offs_data = out_data + ctx_start * st0;
    const int vec_size = x0.numel() - ctx_start * st0; 
    vec_add_f16(x0_offs_data, x1_offs_data, out_offs_data, vec_size);
}

static void add(const Tensor& x0, const Tensor& x1, Tensor& out, const bool last_ctx_only = false)
{
    GTEN_ASSERT(x0.is_2d());
    GTEN_ASSERT(x1.is_2d());
    GTEN_ASSERT(out.is_2d());
    GTEN_ASSERT(x0.shape_eq(x1.shape()));
    GTEN_ASSERT(x0.shape_eq(out.shape()));
    GTEN_ASSERT(x0.dtype() == x1.dtype() && x0.dtype() == out.dtype());

    if (x0.is_quantized())
    {
        add_impl_q8(x0, x1, out, last_ctx_only);
    } else {
        add_impl_f16(x0, x1, out, last_ctx_only);
    }
}

void qk_masked_softmax_f16(const Tensor& q, const Tensor& k, Tensor& qk_out, float scale_factor, const bool last_ctx_only) {
    const Float16* q_data = q.data_ptr<Float16>();
    const Float16* k_data = k.data_ptr<Float16>();
    Float16* out_data = qk_out.data_ptr<Float16>();

    const int q_heads = q.size(0);
    const int n_ctx = q.size(1);
    const int d_head = q.size(2);

    const int qst0 = q.stride(0);
    const int qst1 = q.stride(1);
    const int kst0 = k.stride(0);
    const int kst1 = k.stride(1);
    const int qkst0 = qk_out.stride(0);
    const int qkst1 = qk_out.stride(1);

    const int k_heads = k.size(0);
    const int q_heads_per_group = q_heads / k_heads;

    // q_heads -> 32
    // k_heads -> 4

    // d_head, kv_nhead*dhead, 1
    // kst0: d_head
    // kst1: kv_nhead*dhead
    // h, 0 -> 31
    //  32 / 4 = 8. 

    float* out_buf = g_ops_state.buf(n_ctx);

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;

    for (int h = 0; h < q_heads; h++) {
        for (int qrow = ctx_start; qrow < n_ctx; qrow++) {
            // For each vector in the current head of Q, we only compute the
            // dot_products that are not subsequently masked. That reduces the
            // number of dot products on each head by half.
            const int kcol_max = qrow + 1;
            for (int kcol = 0; kcol < kcol_max; kcol++) {
                const Float16* qrow_data = q_data + (h * qst0 + qrow * qst1);
                const Float16* kcol_data = k_data + ((h / q_heads_per_group) * kst0 + kcol * kst1); // col_data is contigous.
                const float dot_prod = vec_dot_product_f16(qrow_data ,kcol_data, d_head);
                out_buf[kcol] = dot_prod * scale_factor;
            }

            /// Masking operation.
            /// TODO: There is a potential optimization here. Instead of masking by setting
            /// the masked positions to -inf and then computing softmax, we can instead set
            /// the masked positions to zero and skipping computing softmax over masked
            /// positions because they map to zero after softmax is applied over them. This
            /// reduces the number exps and writes by 0.5*n_head*n_ctx*n_ctx.
            const int kcol_start = qrow + 1;
            for (int kcol = kcol_start; kcol < n_ctx; kcol++) {
                out_buf[kcol] = -std::numeric_limits<float>::infinity();  // zero TODO
            }

            // SOFTMAX
            // We use the function sm(x - x_max) = e^(x - x_max) / sum(e^(x - xmax)) instead
            // of the original function sm(x) = e^(x) / sum(e^(x)) because the former is more
            // numerically stable as it prevent overflows in the exponent. The output results
            // is the same in both cases.
            float max = -std::numeric_limits<float>::infinity();

            for (int i = 0; i < n_ctx; i++) {
                const float x = out_buf[i];
                if (x > max)
                    max = x;
            }

            float sum_exp = 0;
            for (int i = 0; i < n_ctx; i++) {
                const float x = out_buf[i];
                const float exp_val = std::exp(x - max);
                out_buf[i] = exp_val;
                sum_exp += exp_val;
            }

            for (int i = 0; i < n_ctx; i++) {
                const float qkw = out_buf[i];
                out_buf[i] = qkw / sum_exp;
            }

            for (int i = 0; i < n_ctx; i++) {
                out_data[h * qkst0 + qrow * qkst1 + i] = fp32_to_fp16(out_buf[i]);
            }
        }
    }
}


void qk_masked_softmax_q8(const Tensor& q, const Tensor& k, Tensor& qk_out, float scale_factor, const bool last_ctx_only)
{
    const Qint8* q_data = q.data_ptr<Qint8>();
    const Qint8* k_data = k.data_ptr<Qint8>();
    Qint8* out_data = qk_out.data_ptr<Qint8>();

    const int q_heads = q.size(0);
    const int n_ctx = q.size(1);
    const int d_head = q.size(2);

    const int qst0 = q.stride(0);
    const int qst1 = q.stride(1);
    const int kst0 = k.stride(0);
    const int kst1 = k.stride(1);
    const int qkst0 = qk_out.stride(0);
    const int qkst1 = qk_out.stride(1);

    const int k_heads = k.size(0);
    const int q_heads_per_group = q_heads / k_heads;

    const Qparams& q_params = q.qparams();
    const Qparams& k_params = k.qparams();
    const int block_size = k_params.block_size();

    GTEN_ASSERT(block_size == q_params.block_size());

    const int blocks_per_head = d_head / block_size;

    float* out_buf = g_ops_state.buf(n_ctx);

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;

    for (int qrow = ctx_start; qrow < n_ctx; qrow++) {
       for (int h = 0; h < q_heads; h++) {
            // `kcol_max` represents number of the dot products that are not subsequently masked.
            const int kcol_max = qrow + 1;
            for (int kcol = 0; kcol < kcol_max; kcol++) {
                
                const Qint8* qrow_data = q_data + (h * qst0 + qrow * qst1);
                const Qint8* kcol_data = k_data + ((h / q_heads_per_group) * kst0 + kcol * kst1); // col_data is contigous.

                // deltas for the row of the current head.
                const Float16* q_delta = q_params.row_deltas(qrow) + h * blocks_per_head;
                const Float16* k_delta = k_params.row_deltas(kcol) + (h / q_heads_per_group) * blocks_per_head;

                const float dot_prod = vec_dot_product_q8(qrow_data, q_delta, kcol_data, k_delta, block_size, d_head);
                out_buf[kcol] = dot_prod * scale_factor;
            }

            /// Masking operation.
            const int kcol_start = qrow + 1;
            for (int kcol = kcol_start; kcol < n_ctx; kcol++) {
                out_buf[kcol] = -std::numeric_limits<float>::infinity();  // zero TODO
            }

            // SOFTMAX
            // Max
            float max = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < n_ctx; i++) {
                const float x = out_buf[i];
                if (x > max)
                    max = x;
            }

            // out[i] = exp(xi - xi_max)
            float sum_exp = 0;
            for (int i = 0; i < n_ctx; i++) {
                const float x = out_buf[i];
                const float exp_val = std::exp(x - max);
                out_buf[i] = exp_val;
                sum_exp += exp_val;
            }

            // out[i] = out[i] / sum[exp(xi - xi_max)]
            for (int i = 0; i < n_ctx; i++) {
                const float qkw = out_buf[i];
                out_buf[i] = qkw / sum_exp;
            }

            const float delta = 1.0f / 127.0f;  // quantization delta where amax=1 because it is softmaxed.
            for (int i = 0; i < n_ctx; i++) {
                out_data[h * qkst0 + qrow * qkst1 + i] = quantize(out_buf[i], delta);
            }
        }
    }
}


void qkv_matmul_q8(const Tensor& qk, const Tensor& v, Tensor& qkv_out, const bool last_ctx_only)
{
    const Qint8* qk_data = qk.data_ptr<Qint8>();
    const Qint8* v_data = v.data_ptr<Qint8>();
    Qint8* out_data = qkv_out.data_ptr<Qint8>();

    const int n_ctx = qk.size(1);
    const int q_heads = qk.size(0);
    
    const int dhead = v.size(1);

    const int v_heads = v.size(0);
    const int q_heads_per_group = q_heads / v_heads;

    const int qkst0 = qk.stride(0);
    const int qkst1 = qk.stride(1);
    const int vst0 = v.stride(0);
    const int vst1 = v.stride(1);
    // qkv shape: [ctx, head, dhead]
    const int qkv_st0 = qkv_out.stride(0);
    const int qkv_st1 = qkv_out.stride(1);

    const float delta = 1.0f / 127.0f;
    // out: [c, h, d] nhead, dhead
    // qkv_st1: dhead
    // v: n_head, d_head, n_ctx

    Qparams& qkv_out_qparams = qkv_out.qparams();
    const Qparams& v_qparams = v.qparams();

    float* buf = g_ops_state.buf(n_ctx + n_ctx + q_heads * dhead);
    float* qk_buf = buf;
    float* v_buf = buf + n_ctx;
    float* out_buf = buf + n_ctx + n_ctx; // qh * dh: dh

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    for (int qkr = ctx_start; qkr < n_ctx; qkr++) {
        for (int h = 0; h < q_heads; h++) {
            const Qint8* qkr_data = qk_data + (h * qkst0 + qkr * qkst1);  // qk_row_data
            
            dequantize_row_scale(qkr_data, delta, qk_buf, n_ctx);

            for (int vc = 0; vc < dhead; vc++) {
                const Qint8* vc_data = v_data + ((h / q_heads_per_group) * vst0 + vc * vst1);

                const int col_idx = (h / q_heads_per_group) * dhead + vc;
                dequantize_col(col_idx, vc_data, v_qparams, v_buf, n_ctx);

                 const float dot_prod = vec_dot_product_f32(qk_buf, v_buf, n_ctx);
                out_buf[h * qkv_st1 + vc] = dot_prod;
            }
        }

        Qint8* out_row_data = out_data + qkr * qkv_st0;
        quantize_row(qkr, out_buf, qkv_out_qparams, out_row_data);
    }
}


void qkv_matmul_f16(const Tensor& qk, const Tensor& v, Tensor& qkv_out, const bool last_ctx_only)
{
    const Float16* qk_data = qk.data_ptr<Float16>();
    const Float16* v_data = v.data_ptr<Float16>();
    Float16* out_data = qkv_out.data_ptr<Float16>();

    const int q_heads = qk.size(0);
    const int n_ctx = qk.size(1);
    const int dhead = v.size(1);

    const int v_heads = v.size(0);
    const int q_heads_per_group = q_heads / v_heads;

    const int qkst0 = qk.stride(0);
    const int qkst1 = qk.stride(1);
    const int vst0 = v.stride(0);
    const int vst1 = v.stride(1);
    // qkv shape: [ctx, head, dhead]
    const int qkv_st0 = qkv_out.stride(0);
    const int qkv_st1 = qkv_out.stride(1);

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    // out: [c, h, d]
    for (int h = 0; h < q_heads; h++) {
        for (int qkr = ctx_start; qkr < n_ctx; qkr++) {
            for (int vc = 0; vc < dhead; vc++) {
                const Float16* qkr_data = qk_data + (h * qkst0 + qkr * qkst1);
                const Float16* vc_data = v_data + ((h / q_heads_per_group) * vst0 + vc * vst1);
                const float dot_prod = vec_dot_product_f16(qkr_data, vc_data, n_ctx);
                out_data[h * qkv_st1 + qkr * qkv_st0 + vc] = fp32_to_fp16(dot_prod);
            }
        }
    }
}


static void qkv_attn_impl(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& qk, Tensor& qkv, int max_ctx, const bool last_ctx)
{
    const int n_ctx = q.size(0);
    const int n_embd = q.size(1);
    const int q_n_head = qk.size(0);
    const int d_head = n_embd / q_n_head;
    const int kv_n_head = k.size(1) / d_head;

    const Tensor q0 = q.view({n_ctx, q_n_head, d_head}).permute({1, 0, 2});
    
    const Tensor k0 = k.view({n_ctx, kv_n_head, d_head}).permute({1, 0, 2});
    // kv_nhead*dhead, d_head, 1
    // d_head, kv_nhead*dhead, 1
    qk.set_strides({max_ctx * max_ctx, max_ctx, 1});


    const float scale_factor = 1.0f / std::sqrt((float)d_head);

    if (q0.is_quantized()) {
        ops::qk_masked_softmax_q8(q0, k0, qk, scale_factor, last_ctx);
    } else {
        ops::qk_masked_softmax_f16(q0, k0, qk, scale_factor, last_ctx);
    }
    

    Tensor v0 = v.view({kv_n_head, d_head, n_ctx});
    v0.set_strides({d_head * max_ctx, max_ctx, 1});

    Tensor qkv0 = qkv.view({n_ctx, q_n_head, d_head});
    
    if (q0.is_quantized()) {
        ops::qkv_matmul_q8(qk, v0, qkv0, last_ctx);
    } else {
        ops::qkv_matmul_f16(qk, v0, qkv0, last_ctx);
    }
}

static void qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& qk, Tensor& qkv, const int max_ctx, const bool last_ctx=false)
{
    const int n_ctx = q.size(0);
    const int n_embd = q.size(1);
    const int n_head = qk.size(0);

    // GTEN_ASSERT(q.is_2d() && q.shape_eq(k.shape()));
    // GTEN_ASSERT(k.is_2d());
    // GTEN_ASSERT(v.is_2d() && v.shape_eq({n_embd, n_ctx}));
    // GTEN_ASSERT(qk.is_3d() && qk.shape_eq({n_head, n_ctx, n_ctx}));
    // GTEN_ASSERT(qkv.is_2d());
    // GTEN_ASSERT(q.dtype() == k.dtype() && k.dtype() == v.dtype() && v.dtype() == qk.dtype() && qk.dtype() == qkv.dtype())
    // GTEN_ASSERT(max_ctx > 0 && max_ctx >= n_ctx);

    qkv_attn_impl(q, k, v, qk, qkv, max_ctx, last_ctx);
}

} // namespace ops
} // namespace gten
