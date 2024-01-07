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
    const size_t max_bufsize = 32 * 1024 * 1024; // 32MB

public:
    OpsState() {
        buf_ = reinterpret_cast<float*>(std::malloc(max_bufsize));
    }
    ~OpsState() { std::free(buf_); }

    // Obtain a ptr to a buffer of size `numel` * sizeof(float).
    float* buf(int numel) const {
        GTEN_ASSERT(numel >= 1 && numel*sizeof(float) <= max_bufsize);
        return buf_;
    }

private:
    float* buf_ = nullptr;
};

static const OpsState g_ops_state = OpsState();


void read_row_to_float(const Tensor& x, const int idx, float* out_buf)
{
    // GTEN_ASSERT(x.is_2d());
    // GTEN_ASSERT idx < n_rows

    switch (x.dtype())
    {
        case kFloat16: {
            const int rowsize = x.dimsize(1);
            const Float16* x_data = x.data_ptr<Float16>() + idx * rowsize;
            for (int i = 0; i < rowsize; i++) {
                out_buf[i] = fp16_to_fp32(x_data[i]);
            }
        } break;
        case kQint8: {
            const int rowsize = x.dimsize(1);
            const Q8Block* x_data = x.data_ptr<Q8Block>() + idx * rowsize/globs::q8_block_size;
            dequantize_row(x_data, out_buf, rowsize);
        } break;
        default: {
            GTEN_ASSERT(false);
        } break;
    }
}


void write_row_from_float(float* x, const int idx, Tensor& out)
{
    switch (out.dtype())
    {
        case kFloat16: {
            const int rowsize = out.dimsize(1);
            Float16* out_data = out.data_ptr<Float16>() + idx * rowsize;
            for (int i = 0; i < rowsize; i++) {
                out_data[i] = fp32_to_fp16(x[i]);
            }
        } break;
        case kQint8: {
            const int rowsize = out.dimsize(1);
            Q8Block* out_data = out.data_ptr<Q8Block>() + idx * rowsize/globs::q8_block_size;
            quantize_row(x, out_data, rowsize);
           
        } break;
        default: {
            GTEN_ASSERT(false);
            break;
        }
    }
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

#else
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


static float vec_dot_product_q8(const Q8Block* inp0, const Q8Block* inp1, const int vec_size)
{
    // GTEN_ASSERTM(vec_size % blk_size == 0, "row size: %d is incompatible with block size: %d", vec_size, blk_size);

    const int block_size = globs::q8_block_size;
    GTEN_ASSERT(vec_size % block_size == 0);
    const int n_blocks = vec_size / block_size;

#ifdef GTEN_SIMD_AVX
    GTEN_ASSERT(block_size % 16 == 0);
    // Dot product accumulator with 4 slots. The sum of the four accumulators gives the
    // dot product.
    __m128 dot_accum = _mm_set1_ps(0.0f);

    for (int i = 0; i < n_blocks; i++)
    {
        const Q8Block* b0 = inp0 + i;
        const Q8Block* b1 = inp1 + i;

        // const __m128 a_blk_delta = _mm_broadcast_ss(a_ds + i);
        // const __m128 b_blk_delta = _mm_broadcast_ss(b_ds + i); 
        // const __m128 ab_blk_delta = _mm_mul_ps(a_blk_delta, b_blk_delta);
        const __m128 block_delta_multiplier = _mm_set1_ps(fp16_to_fp32(b0->delta) * fp16_to_fp32(b1->delta));

        // dotprod = aq0*ad * bq0*bd + aq1*ad * bq1*bd + ... + aqN*ad + bqN*bd
        //         = adbd(aq0 * bq0) + adbd(aq1 * bq1) + ... + adbd(aqN * bqN)
        //         = adbd(aq0 * bq0 + aq1 * bq1 + ... + aqN * bqN)
        // We compute integer arithmetic inside the brackets and scale by the block
        // quantisation deltas.

        // Integer dot product accumulator for current block.
        __m128i blk_dot_accum = _mm_set1_epi32(0);

        for (int j = 0; j < block_size; j += 16)
        {
            // Load 64-bit(8 1-byte quants) in the lower half. [8-quants, -------].
            const Qint8* b0_data = b0->data + j;
            const Qint8* b1_data = b1->data + j;

            const __m128i a00 = _mm_loadu_si64(b0_data);
            const __m128i a01 = _mm_loadu_si64(b0_data + 8);

            const __m128i b00 = _mm_loadu_si64(b1_data);
            const __m128i b01 = _mm_loadu_si64(b1_data + 8);

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

        const __m128 blk_dot_accum_f = _mm_cvtepi32_ps(blk_dot_accum);
        dot_accum = _mm_add_ps(dot_accum, _mm_mul_ps(blk_dot_accum_f, block_delta_multiplier));
    }

    const __m128 dotsum0 = _mm_hadd_ps(dot_accum, dot_accum);
    const __m128 dotsum1 = _mm_hadd_ps(dotsum0, dotsum0);
    const float dot_prod = _mm_cvtss_f32(dotsum1);

#else

    float dot_prod = 0.0f;

    for (int i = 0; i < n_blocks; i++)
    {
        const Q8Block* b0 = inp0 + i;
        const Q8Block* b1 = inp1 + i;

        int block_dot_prod = 0;
        for (int j = 0; j < block_size; j++)
        {
            block_dot_prod += b0->data[j] * b1->data[j];
        }

        const float b0_delta = fp16_to_fp32(b0->delta);
        const float b1_delta = fp16_to_fp32(b1->delta);
        dot_prod += block_dot_prod * b0_delta * b1_delta;
    }
#endif

    return dot_prod;
}

static void copy_row(const Tensor& src, Tensor& dest, const int src_row_idx, const int dest_row_idx)
{
    switch(src.dtype())
    {
        case kFloat16:
        {
            const int rowsize = src.dimsize(1);
            const Float16* src_data = src.data_ptr<Float16>() + src_row_idx * rowsize;
            Float16* dest_data = dest.data_ptr<Float16>() + dest_row_idx * rowsize;
            std::memcpy(dest_data, src_data, rowsize * sizeof(Float16));
        } break;
        case kQint8:
        {
            const int rowsize = src.dimsize(1);
            GTEN_ASSERT(rowsize % globs::q8_block_size == 0);
            const int n_blocks = rowsize / globs::q8_block_size;
            const Q8Block* src_data = src.data_ptr<Q8Block>() + src_row_idx * n_blocks;
            Q8Block* dest_data = dest.data_ptr<Q8Block>() + dest_row_idx * n_blocks;
            std::memcpy(dest_data, src_data, n_blocks * sizeof(Q8Block));
        }  break;
        default: {
            GTEN_ASSERT(false);
        } break;
    }
}


static void tensor_row_index_impl(const Tensor& src, const Tensor& indices, Tensor& out, const int start_pos)
{
    const int32_t* indices_data = indices.data_ptr<int32_t>();

    const int n_ctx = indices.numel();
    for (int i = start_pos; i < n_ctx; i++) {
        const int src_row_idx = indices_data[i];
        const int dest_row_idx = i;
        copy_row(src, out, src_row_idx, dest_row_idx);
    }
}

/// @brief Copies the indexed rows of the source tensor to output tensor.
/// @param src A 2-d tensor to be indexed.
/// @param indices A 1-d tensor of indices with dtype = int.
/// @param out A 2d tensor with enough capacity to fit the indexed rows. Its dtype
///  must be the same as source tensor.
/// @param last_token_only Whether to index the last token only, if others are cached.
void token_embed(const Tensor& weight, const Tensor& tokens, Tensor& out, const int start_pos = 0)
{
    GTEN_ASSERT(weight.is_2d());
    GTEN_ASSERT(tokens.is_1d() && tokens.dtype() == kInt32);
    const int n_ctx = tokens.dimsize(0);
    const int n_embd = weight.dimsize(1);
    GTEN_ASSERT(out.shape_eq({n_ctx, n_embd}));
    GTEN_ASSERT(weight.dtype() == out.dtype());

    tensor_row_index_impl(weight, tokens, out, start_pos);
}

static void emb_matmul_impl_q8(const Tensor& x, const Tensor& w, Tensor& out)
{
    const Q8Block* x_data = x.data_ptr<Q8Block>();
    const Q8Block* w_data = w.data_ptr<Q8Block>();
    float* out_data = out.data_ptr<float>();

    const int n_ctx = x.dimsize(0);
    const int n_embd = x.dimsize(1);
    const int n_vocab = w.dimsize(0);

#ifdef GTEN_OMP
    #pragma omp parallel for collapse(2)
#endif
    for (int xrow = n_ctx-1; xrow < n_ctx; xrow++) {
        for (int wrow = 0; wrow < n_vocab; wrow++) {
            const Q8Block* x_row_data = x_data + xrow * n_embd / globs::q8_block_size;
            const Q8Block* w_row_data = w_data + wrow * n_embd / globs::q8_block_size;

            const float dot_prod = vec_dot_product_q8(x_row_data, w_row_data, n_embd);
            out_data[wrow] = dot_prod;
        }
    }
}

static void emb_matmul_impl_f16(const Tensor& x, const Tensor& w, Tensor& out)
{
    const Float16* x_data = x.data_ptr<Float16>();
    const Float16* w_data = w.data_ptr<Float16>();
    float* out_data = out.data_ptr<float>();

    const int n_ctx = x.dimsize(0);
    const int n_embd = x.dimsize(1);
    const int n_vocab = w.dimsize(0);

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
    const int n_embd = x.dimsize(1);
    GTEN_ASSERT(weight.is_2d() && weight.dimsize(1) == n_embd);
    const int n_vocab = weight.dimsize(0);
    GTEN_ASSERT(out.is_1d() && out.dtype() == kFloat32 && out.dimsize(0) == n_vocab);

    if (weight.is_quantized()) {
        emb_matmul_impl_q8(x, weight, out);
    } else {
        emb_matmul_impl_f16(x, weight, out);
    }
}

// matmul
// add

static void matmul_2d_impl_q8(const Tensor& x, const Tensor& w, Tensor& out, const int start_pos)
{
    const Q8Block* x_data = x.data_ptr<Q8Block>();
    const Q8Block* w_data = w.data_ptr<Q8Block>();
    Q8Block* out_data = out.data_ptr<Q8Block>();

    const int n_ctx = x.dimsize(0);
    const int n_embd = x.dimsize(1);
    const int d_out = w.dimsize(0);
    const int x_st0 = x.stride(0);
    const int w_st0 = w.stride(0);
    const int out_st0 = out.stride(0);

    float* out_buf = g_ops_state.buf(d_out);

    for (int xrow = start_pos; xrow < n_ctx; xrow++) {
        for (int wrow = 0; wrow < d_out; wrow++) {
            const Q8Block* xrow_data = x_data + xrow * x_st0 / globs::q8_block_size;
            const Q8Block* wrow_data = w_data + wrow * w_st0 / globs::q8_block_size;

            const float dot_prod = vec_dot_product_q8(xrow_data, wrow_data, n_embd);

            out_buf[wrow] = dot_prod;
        }

        Q8Block* outrow_data = out_data + xrow * out_st0 / globs::q8_block_size;
        quantize_row(out_buf, outrow_data, d_out);
    }
}


// dot_product: readline
static void matmul_2d_impl_f16(const Tensor& x, const Tensor& w, Tensor& out, const int start_pos)
{
    const Float16* x_data = x.data_ptr<Float16>();
    const Float16* w_data = w.data_ptr<Float16>();
    Float16* out_data = out.data_ptr<Float16>();

    const int n_ctx = x.dimsize(0);
    const int n_embd = x.dimsize(1);
    const int d_out = w.dimsize(0);
    const int x_st0 = x.stride(0);
    const int w_st0 = w.stride(0);
    const int out_st0 = out.stride(0); 

// #ifdef GTEN_OMP
//     #pragma omp parallel for collapse(2)
// #endif
    for (int xrow = start_pos; xrow < n_ctx; xrow++) {
        for (int wrow = 0; wrow < d_out; wrow++) {
            const Float16* xrow_data = x_data + xrow * x_st0;
            const Float16* wrow_data = w_data + wrow * w_st0;
            float dot_prod = vec_dot_product_f16(xrow_data, wrow_data, n_embd);
            
            out_data[xrow * out_st0 + wrow] = fp32_to_fp16(dot_prod); 
        }
    }
}


static void matmul_2d(const Tensor& x, const Tensor& w, Tensor& out, const int start_pos=0)
{
    const int n_ctx = x.dimsize(0);
    const int n_out = w.dimsize(0);
    const int n_embd = x.dimsize(1);

    GTEN_ASSERT(x.is_2d());
    GTEN_ASSERT(w.is_2d() && w.dimsize(1) == n_embd);
    GTEN_ASSERT(x.dtype() == w.dtype() && w.dtype() == out.dtype());
    GTEN_ASSERT(out.is_2d() && out.shape_eq({n_ctx, n_out}));

    if (x.is_quantized()) {
        matmul_2d_impl_q8(x, w, out, start_pos);
    } else {
        matmul_2d_impl_f16(x, w, out, start_pos);
    }
}


static void silu_impl(const Tensor& inp, Tensor& out, const int start_pos)
{
    const int n_ctx = inp.dimsize(0);
    const int n_embd = inp.dimsize(1);

    float* buf = g_ops_state.buf(n_embd);

    for (int i = start_pos; i < n_ctx; i++)
    {
        read_row_to_float(inp, i, buf);

        for (int j = 0; j < n_embd; j++) {
            const float x = buf[j];
            buf[j] = x / (1.0f + std::exp(-x));
        }

        write_row_from_float(buf, i, out);
    }
}


static void silu(const Tensor& inp, Tensor& out, const int start_pos=0)
{
    GTEN_ASSERT(inp.shape_eq(out.shape()));
    GTEN_ASSERT(inp.dtype() == out.dtype());

    silu_impl(inp, out, start_pos);
}

static void silu_inplace(Tensor& inp, const int start_pos=0)
{
    silu_impl(inp, inp, start_pos);
}


static void rotary_emb_impl(Tensor& inp, const int d_head, const int start_pos)
{
    const int n_ctx = inp.dimsize(0);
    const int n_embd = inp.dimsize(1);
    const int n_head = n_embd / d_head;

    Tensor inpx = inp.view({n_ctx, n_head, d_head});
    const int st0 = inpx.stride(0);

    float* inp_buf = g_ops_state.buf(n_embd);

    const float d = static_cast<float>(d_head);
    for (int i = start_pos; i < n_ctx; ++i) {
        read_row_to_float(inp, i, inp_buf);

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

        write_row_from_float(inp_buf, i, inp);
    }
}

static void rotary_emb(Tensor& inp, const int d_head, const int start_pos=0)
{
    rotary_emb_impl(inp, d_head, start_pos);
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

static void rms_norm_impl(const Tensor& inp, const Tensor& weight, Tensor& out, const int start_pos)
{
    const int n_ctx = inp.dimsize(0);
    const int n_embd = inp.dimsize(1);

    const Float16* weight_data = weight.data_ptr<Float16>();
    float* inp_buf = g_ops_state.buf(n_embd * 2);
    float* out_buf = inp_buf + n_embd;

    for (int i = start_pos; i < n_ctx; i++)
    {
        read_row_to_float(inp, i, inp_buf);

        rms_norm_vec_f32(inp_buf, weight_data, out_buf, n_embd);

        write_row_from_float(out_buf, i, out);
    }
}

static void rms_norm(const Tensor& inp, const Tensor& weight, Tensor& out, const int start_pos=0) {
    const int n_embd = inp.dimsize(1);
    GTEN_ASSERT(weight.dimsize(0) == n_embd);
    GTEN_ASSERT(inp.is_2d() && inp.dtype() == out.dtype());
    GTEN_ASSERT(weight.is_1d());
    GTEN_ASSERT(inp.shape_eq(out.shape()));

    rms_norm_impl(inp, weight, out, start_pos);
}

static void vec_mul_f32(const float* a, const float* b, float* out, int vec_size) {
    for (int i = 0; i < vec_size; i++) {
        out[i] = a[i] * b[i];
    }
}

static void mul_impl(const Tensor& x0, const Tensor& x1, Tensor& out, const int start_pos)
{
    const int n_ctx = x0.dimsize(0);
    const int n_embd = x1.dimsize(1);

    float* x0_buf = g_ops_state.buf(n_embd * 3);
    float* x1_buf = x0_buf + n_embd;
    float* out_buf = x1_buf + n_embd;

    for (int i = start_pos; i < n_ctx; i++)
    {
        read_row_to_float(x0, i, x0_buf);
        read_row_to_float(x1, i, x1_buf);

        vec_mul_f32(x0_buf, x1_buf, out_buf, n_embd);

        write_row_from_float(out_buf, i, out);
    }
}


static void mul(const Tensor& inp0, const Tensor& inp1, Tensor& out, const int start_pos=0)
{
    GTEN_ASSERT(inp0.dtype() == inp1.dtype() && inp1.dtype() == out.dtype());
    GTEN_ASSERT(inp0.shape_eq(inp1.shape()) && inp1.shape_eq(out.shape()));

    mul_impl(inp0, inp1, out, start_pos);
}

static void mul_inplace(Tensor& inp0, const Tensor& inp1, const int start_pos=0)
{
    GTEN_ASSERT(inp0.dtype() == inp1.dtype());
    GTEN_ASSERT(inp0.shape_eq(inp1.shape()));

    mul_impl(inp0, inp1, inp0, start_pos);
}


static void add_impl(const Tensor& x0, const Tensor& x1, Tensor& out, const int start_pos)
{
    // tensor.row_to_float
    const int n_ctx = x0.dimsize(0);
    const int n_embd = x0.dimsize(1);

    float* x0_buf = g_ops_state.buf(n_embd * 3);
    float* x1_buf = x0_buf + n_embd;
    float* out_buf = x1_buf + n_embd;

    for (int i = start_pos; i < n_ctx; i++)
    {
        read_row_to_float(x0, i, x0_buf);
        read_row_to_float(x1, i, x1_buf);

        vec_add_f32(x0_buf, x1_buf, out_buf, n_embd);

        write_row_from_float(out_buf, i, out);
    }
}

static void add(const Tensor& x0, const Tensor& x1, Tensor& out, const int start_pos=0)
{
    GTEN_ASSERT(x0.is_2d());
    GTEN_ASSERT(x1.is_2d());
    GTEN_ASSERT(out.is_2d());
    GTEN_ASSERT(x0.shape_eq(x1.shape()));
    GTEN_ASSERT(x0.shape_eq(out.shape()));
    GTEN_ASSERT(x0.dtype() == x1.dtype() && x0.dtype() == out.dtype());

    add_impl(x0, x1, out, start_pos);
}

void qk_masked_softmax_f16(const Tensor& q, const Tensor& k, Tensor& qk_out, float scale_factor, const int start_pos) {
    const Float16* q_data = q.data_ptr<Float16>();
    const Float16* k_data = k.data_ptr<Float16>();
    Float16* out_data = qk_out.data_ptr<Float16>();

    const int q_heads = q.dimsize(0);
    const int n_ctx = q.dimsize(1);
    const int d_head = q.dimsize(2);

    const int qst0 = q.stride(0);
    const int qst1 = q.stride(1);
    const int kst0 = k.stride(0);
    const int kst1 = k.stride(1);
    const int qkst0 = qk_out.stride(0);
    const int qkst1 = qk_out.stride(1);

    const int k_heads = k.dimsize(0);
    const int q_heads_per_group = q_heads / k_heads;

    // q_heads -> 32
    // k_heads -> 4

    float* out_buf = g_ops_state.buf(n_ctx);

    for (int h = 0; h < q_heads; h++) {
        for (int qrow = start_pos; qrow < n_ctx; qrow++) {
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


void qk_masked_softmax_q8(const Tensor& q, const Tensor& k, Tensor& qk_out, float scale_factor, const int start_pos)
{
    const Q8Block* q_data = q.data_ptr<Q8Block>();
    const Q8Block* k_data = k.data_ptr<Q8Block>();
    Qint8* out_data = qk_out.data_ptr<Qint8>();

    const int q_heads = q.dimsize(0);
    const int n_ctx = q.dimsize(1);
    const int d_head = q.dimsize(2);

    const int qst0 = q.stride(0);
    const int qst1 = q.stride(1);
    const int kst0 = k.stride(0);
    const int kst1 = k.stride(1);
    const int qkst0 = qk_out.stride(0);
    const int qkst1 = qk_out.stride(1);

    const int k_heads = k.dimsize(0);
    const int q_heads_per_group = q_heads / k_heads;

    const int block_size = globs::q8_block_size;

    const int blocks_per_head = d_head / block_size;

    float* out_buf = g_ops_state.buf(n_ctx);

    for (int qrow = start_pos; qrow < n_ctx; qrow++) {
       for (int h = 0; h < q_heads; h++) {
            // `kcol_max` represents number of the dot products that are not subsequently masked.
            const int kcol_max = qrow + 1;
            for (int kcol = 0; kcol < kcol_max; kcol++) {
                const Q8Block* qrow_data = q_data + (h * qst0/block_size + qrow * qst1/block_size);
                const Q8Block* kcol_data = k_data + ((h / q_heads_per_group) * kst0/block_size + kcol * kst1/block_size); // col_data is contigous.

                const float dot_prod = vec_dot_product_q8(qrow_data, kcol_data, d_head);
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
                out_data[h * qkst0 + qrow * qkst1 + i] = quantize_single(out_buf[i], delta);
            }
        }
    }
}


void qkv_matmul_q8(const Tensor& qk, const Tensor& v, Tensor& qkv_out, const int start_pos)
{
    const Qint8* qk_data = qk.data_ptr<Qint8>();
    const Q8Block* v_data = v.data_ptr<Q8Block>();
    Q8Block* out_data = qkv_out.data_ptr<Q8Block>();

    const int n_ctx = qk.dimsize(1);
    const int q_heads = qk.dimsize(0);
    
    const int dhead = v.dimsize(1);

    const int v_heads = v.dimsize(0);
    const int q_heads_per_group = q_heads / v_heads;

    const int qkst0 = qk.stride(0);
    const int qkst1 = qk.stride(1);
    const int vst0 = v.stride(0);
    const int vst1 = v.stride(1);
    const int vst2 = v.stride(2);
    // qkv shape: [ctx, head, dhead]
    const int qkv_st0 = qkv_out.stride(0);
    const int qkv_st1 = qkv_out.stride(1);

    const float delta = 1.0f / 127.0f;
    // out: [c, h, d] nhead, dhead
    // qkv_st1: dhead
    // v: n_head, d_head, n_ctx

    const int v_n_embd = v_heads*dhead;
    float* qk_row_buf = g_ops_state.buf(n_ctx + n_ctx*v_n_embd + q_heads*dhead);
    float* v_buf = qk_row_buf + n_ctx;
    float* out_buf = v_buf + n_ctx*v_n_embd; // qh * dh: dh

    // Dequantize v and transpose it.
    const int n_blocks = v_n_embd / globs::q8_block_size;

    for (int i = 0; i < n_ctx; i++) { // xrow
        for (int j = 0; j < n_blocks; j++) {
            const Q8Block* blk = v_data + i * n_blocks + j;
            const float block_delta = fp16_to_fp32(blk->delta);
            
            for (int k = 0; k < globs::q8_block_size; k++) {
                const int col_idx = j * globs::q8_block_size + k;
                v_buf[i + col_idx * n_ctx] = dequantize_single(blk->data[k], block_delta);
            }
        }   
    }
    

    for (int qkr = start_pos; qkr < n_ctx; qkr++) {
        for (int h = 0; h < q_heads; h++) {
            const Qint8* qkr_data = qk_data + (h * qkst0 + qkr * qkst1);  // qk_row_data
            
            dequantize_row(qkr_data, delta, qk_row_buf, n_ctx);

            for (int vc = 0; vc < dhead; vc++) {
                const float* v_col_buf = v_buf + ((h / q_heads_per_group) * dhead*n_ctx + vc * n_ctx);

                const float dot_prod = vec_dot_product_f32(qk_row_buf, v_col_buf, n_ctx);
                out_buf[h*dhead + vc] = dot_prod;
            }
        }

        Q8Block* out_row_data = out_data + qkr * qkv_st0 / globs::q8_block_size;
        quantize_row(out_buf, out_row_data,  q_heads*dhead);
        
    }

}


void qkv_matmul_f16(const Tensor& qk, const Tensor& v, Tensor& qkv_out, const int start_pos)
{
    const Float16* qk_data = qk.data_ptr<Float16>();
    const Float16* v_data = v.data_ptr<Float16>();
    Float16* out_data = qkv_out.data_ptr<Float16>();

    const int q_heads = qk.dimsize(0);
    const int n_ctx = qk.dimsize(1);
    const int dhead = v.dimsize(1);

    const int v_heads = v.dimsize(0);
    const int q_heads_per_group = q_heads / v_heads;

    const int qkst0 = qk.stride(0);
    const int qkst1 = qk.stride(1);
    const int vst0 = v.stride(0);
    const int vst1 = v.stride(1);
    const int vst2 = v.stride(2);
    // qkv shape: [ctx, head, dhead]
    const int qkv_st0 = qkv_out.stride(0);
    const int qkv_st1 = qkv_out.stride(1);

    // out: [c, h, d]
    for (int h = 0; h < q_heads; h++) {
        for (int qkr = start_pos; qkr < n_ctx; qkr++) {
            for (int vc = 0; vc < dhead; vc++) {

                float dot_prod = 0.0f;
                for (int i = 0; i < n_ctx; i++) {
                    const int qk_idx = h * qkst0 + qkr * qkst1 + i;
                    const int v_idx = (h / q_heads_per_group) * vst0 + vc * vst1 + i * vst2;

                    const float qx = fp16_to_fp32(qk_data[qk_idx]);
                    const float vx = fp16_to_fp32(v_data[v_idx]);
                    dot_prod += qx * vx;
                }

                out_data[h * qkv_st1 + qkr * qkv_st0 + vc] = fp32_to_fp16(dot_prod);
            }
        }
    }
}

// [n_ctx, n_head, d_head]     -> [n_head, d_head, n_ctx]
// [n_head*d_head, d_head, 1]     [d_head, 1, n_head*d_head]

static void qkv_attn_impl(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& qk, Tensor& qkv, int max_ctx, const int start_pos)
{
    const int n_ctx = q.dimsize(0);
    const int n_embd = q.dimsize(1);
    const int q_n_head = qk.dimsize(0);
    const int d_head = n_embd / q_n_head;
    const int kv_n_head = k.dimsize(1) / d_head;

    const Tensor q0 = q.view({n_ctx, q_n_head, d_head}).permute({1, 0, 2});
    
    const Tensor k0 = k.view({n_ctx, kv_n_head, d_head}).permute({1, 0, 2});
    // kv_nhead*dhead, d_head, 1
    // d_head, kv_nhead*dhead, 1
    qk.set_strides({max_ctx * max_ctx, max_ctx, 1});


    const float scale_factor = 1.0f / std::sqrt((float)d_head);

    if (q0.is_quantized()) {
        ops::qk_masked_softmax_q8(q0, k0, qk, scale_factor, start_pos);
    } else {
        ops::qk_masked_softmax_f16(q0, k0, qk, scale_factor, start_pos);
    }
    

    // Tensor v0 = v.view({kv_n_head, d_head, n_ctx});
    // v0.set_strides({d_head * max_ctx, max_ctx, 1});

    Tensor v0 = v.view({n_ctx, kv_n_head, d_head}).permute({1, 2, 0});

    Tensor qkv0 = qkv.view({n_ctx, q_n_head, d_head});
    
    if (q0.is_quantized()) {
        ops::qkv_matmul_q8(qk, v0, qkv0, start_pos);
    } else {
        ops::qkv_matmul_f16(qk, v0, qkv0, start_pos);
    }
}

static void qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& qk, Tensor& qkv, const int max_ctx, const int start_pos=0)
{
    const int n_ctx = q.dimsize(0);
    const int n_embd = q.dimsize(1);
    const int n_head = qk.dimsize(0);

    GTEN_ASSERT(q.is_2d());
    GTEN_ASSERT(k.is_2d());
    GTEN_ASSERT(v.is_2d());
    GTEN_ASSERT(qk.is_3d() && qk.shape_eq({n_head, n_ctx, n_ctx}));
    GTEN_ASSERT(qkv.is_2d() && qkv.shape_eq({n_ctx, n_embd}));
    GTEN_ASSERT(q.dtype() == k.dtype() && k.dtype() == v.dtype() && v.dtype() == qk.dtype() && qk.dtype() == qkv.dtype())
    GTEN_ASSERT(max_ctx > 0 && max_ctx >= n_ctx);

    qkv_attn_impl(q, k, v, qk, qkv, max_ctx, start_pos);
}

} // namespace ops
} // namespace gten
