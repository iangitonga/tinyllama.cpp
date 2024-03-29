#include <cstring>

#include "log.h"
#include "quants.h"
#include "tensor.h"
#include "simd_ops.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
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


void read_row_to_float(const char* inp, Dtype inp_dtype, float* out_buf, const int rowsize)
{
    switch (inp_dtype)
    {
        case kQint4: 
        {
            const Q4Block* inp_data = reinterpret_cast<const Q4Block*>(inp);
            q4_dequantize_row(inp_data, out_buf, rowsize);
        } break;
        case kQint8:
        {
            const Q8Block* inp_data = reinterpret_cast<const Q8Block*>(inp);
            q8_dequantize_row(inp_data, out_buf, rowsize);
        } break;
        case kFloat16:
        {
            const Float16* inp_data = reinterpret_cast<const Float16*>(inp);
            for (int i = 0; i < rowsize; i++) {
                out_buf[i] = fp16_to_fp32(inp_data[i]);
            }
        } break;
        case kFloat32:
        {
            std::memcpy(out_buf, inp, rowsize*sizeof(float));
        } break;
        default:
        {
            GTEN_ASSERT(false);
        } break;
    }
}


void write_row_from_float(float* inp, char* out, Dtype out_dtype, int rowsize) {
    switch (out_dtype) {
        case kQint8:
        {
            Q8Block* out_data = reinterpret_cast<Q8Block*>(out);
            q8_quantize_row(inp, out_data, rowsize);
        } break;
        case kFloat16:
        {
            Float16* out_data = reinterpret_cast<Float16*>(out);
            for (int i = 0; i < rowsize; i++) {
                out_data[i] = fp32_to_fp16(inp[i]);
            }
        } break;
        case kFloat32:
        {
            std::memcpy(out, inp, rowsize*sizeof(float));
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

#if defined(__AVX__)
    GTEN_ASSERT(block_size % 16 == 0);
    // Dot product accumulator with 4 slots. The sum of the four accumulators gives the
    // dot product.
    __m128 dot_accum = _mm_set1_ps(0.0f);

    for (int i = 0; i < n_blocks; i++)
    {
        const Q8Block* b0 = inp0 + i;
        const Q8Block* b1 = inp1 + i;

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
        // const __m128 a_blk_delta = _mm_broadcast_ss(a_ds + i);
        // const __m128 b_blk_delta = _mm_broadcast_ss(b_ds + i); 
        // const __m128 ab_blk_delta = _mm_mul_ps(a_blk_delta, b_blk_delta);
        const __m128 block_delta_multiplier = _mm_set1_ps(fp16_to_fp32(b0->delta) * fp16_to_fp32(b1->delta));
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


static float vec_dot_product_q8_q4(const Q8Block* inp0, const Q4Block* inp1, const int vec_size)
{
    const int block_size = globs::q8_block_size;
    GTEN_ASSERT(block_size == globs::q4_block_size && vec_size % block_size == 0);
    const int n_blocks = vec_size / block_size;

#if defined(__AVX__)
    GTEN_ASSERT(block_size % 16 == 0);
    // Dot product accumulator with 4 slots. The sum of the four accumulators gives the
    // dot product.
    __m128 dot_accum = _mm_set1_ps(0.0f);

    for (int i = 0; i < n_blocks; i++)
    {
        const Q8Block* a0 = inp0 + i;
        const Q4Block* b0 = inp1 + i;

        // Integer dot product accumulator for current block.
        __m128i blk_dot_accum = _mm_set1_epi32(0);

        const Qint8* a0_data = a0->data;
        const __m128i a00 = _mm_loadu_si64(a0_data);
        const __m128i a01 = _mm_loadu_si64(a0_data + 8);
        const __m128i a02 = _mm_loadu_si64(a0_data + 16);
        const __m128i a03 = _mm_loadu_si64(a0_data + 24);

        // Convert 8 quants in the lower half to 16-bit ints.
        const __m128i a04 = _mm_cvtepi8_epi16(a00);
        const __m128i a05 = _mm_cvtepi8_epi16(a01);
        const __m128i a06 = _mm_cvtepi8_epi16(a02);
        const __m128i a07 = _mm_cvtepi8_epi16(a03);

        // load 16 bytes => 
        // load into lower 64 bits.
        const __m128i b00 = _mm_loadu_si64(b0->data);
        const __m128i b01 = _mm_loadu_si64(b0->data + 8);

        const __m128i b02 = _mm_cvtepu8_epi16(b00);
        const __m128i b03 = _mm_cvtepu8_epi16(b01);

        const __m128i add_vec = _mm_set1_epi16(-7);
        const __m128i b04 = _mm_add_epi16(_mm_srli_epi16(b02, 4), add_vec);
        const __m128i b05 = _mm_add_epi16(_mm_srli_epi16(b03, 4), add_vec);

        const __m128i and_vec = _mm_set1_epi16(0b0000000000001111);
        const __m128i b06 = _mm_add_epi16(_mm_and_si128(b02, and_vec), add_vec);
        const __m128i b07 = _mm_add_epi16(_mm_and_si128(b03, and_vec), add_vec);

        // Multiply the 8 16-bit ints to obtain 8 32-bit ints and add adjacent
        // values to obtain 4 32-bit ints.
        // TODO: Can we instead do 16-bit to 16-bit e.g _mullo_epi16
        const __m128i c00 = _mm_madd_epi16(a04, b04);
        const __m128i c01 = _mm_madd_epi16(a05, b05);
        const __m128i c02 = _mm_madd_epi16(a06, b06);
        const __m128i c03 = _mm_madd_epi16(a07, b07);

        // Add the results and add the output to the accumulator.
        const __m128i c05 = _mm_add_epi32(c00, c01);
        const __m128i c06 = _mm_add_epi32(c05, _mm_add_epi32(c02, c03));
        blk_dot_accum = _mm_add_epi32(blk_dot_accum, c06);


        const __m128 blk_dot_accum_f = _mm_cvtepi32_ps(blk_dot_accum);
        // const __m128 a_blk_delta = _mm_broadcast_ss(a_ds + i);
        // const __m128 b_blk_delta = _mm_broadcast_ss(b_ds + i); 
        // const __m128 ab_blk_delta = _mm_mul_ps(a_blk_delta, b_blk_delta);
        const __m128 block_delta_multiplier = _mm_set1_ps(fp16_to_fp32(a0->delta) * fp16_to_fp32(b0->delta));
        dot_accum = _mm_add_ps(dot_accum, _mm_mul_ps(blk_dot_accum_f, block_delta_multiplier));
    }

    const __m128 dotsum0 = _mm_hadd_ps(dot_accum, dot_accum);
    const __m128 dotsum1 = _mm_hadd_ps(dotsum0, dotsum0);
    const float dot_prod = _mm_cvtss_f32(dotsum1);

#elif defined(__ARM_NEON)

    float32x4_t dot_accum = vdupq_n_f32(0.0f);

    for (int i = 0; i < n_blocks; i++)
    {
        const Q8Block* a0 = inp0 + i;
        const Q4Block* b0 = inp1 + i;

        // Load Q-8 block.
        const Qint8* a0_data = a0->data;
        const int8x8_t a00 = vld1_s8(a0_data);
        const int8x8_t a01 = vld1_s8(a0_data + 8);
        const int8x8_t a02 = vld1_s8(a0_data + 16);
        const int8x8_t a03 = vld1_s8(a0_data + 24);

        // Load Q-4 block.
        const Qint4* b0_data = b0->data;
        const uint8x8_t b00 = vld1_u8(b0_data);
        const uint8x8_t b01 = vld1_u8(b0_data + 8);

        // Extract first 16 quants.
        const int8x8_t b00_high = vreinterpret_s8_u8(vshr_n_u8(b00, 4));
        const int8x8_t b01_high = vreinterpret_s8_u8(vshr_n_u8(b01, 4));
        const int8x8_t add_vec = vdup_n_s8(-7);
        const int8x8_t b04 = vadd_s8(b00_high, add_vec);
        const int8x8_t b05 = vadd_s8(b01_high, add_vec);

        // Extract last 16 quants.
        const uint8x8_t and_vec = vdup_n_u8(0b00001111);
        const int8x8_t b00_low = vreinterpret_s8_u8(vand_u8(b00, and_vec));
        const int8x8_t b01_low = vreinterpret_s8_u8(vand_u8(b01, and_vec));
        const int8x8_t b06 = vadd_s8(b00_low, add_vec);
        const int8x8_t b07 = vadd_s8(b01_low, add_vec);

        const int16x8_t c00 = vmull_s8(a00, b04);
        const int16x8_t c01 = vmull_s8(a01, b05);
        const int16x8_t c02 = vmull_s8(a02, b06);
        const int16x8_t c03 = vmull_s8(a03, b07);

        const int16x8_t c04 = vaddq_s16(c00, c01);
        const int16x8_t c05 = vaddq_s16(c02, c03);

        const int16x8_t c06 = vaddq_s16(c04, c05);
        const int16x4_t c06_low = vget_low_s16(c06);
        const int16x4_t c06_high = vget_high_s16(c06);

        const int32x4_t c07 = vaddl_s16(c06_low, c06_high);
        const float32x4_t c08 = vcvtq_f32_s32(c07);

        const float32x4_t blk_delta_multiplier = vdupq_n_f32(fp16_to_fp32(a0->delta) * fp16_to_fp32(b0->delta));
        dot_accum = vaddq_f32(dot_accum, vmulq_f32(c08, blk_delta_multiplier));
    } 

    const float32x2_t dot_accum_temp = vadd_f32(vget_low_f32(dot_accum), vget_high_f32(dot_accum));
    const float dot_prod = vget_lane_f32(dot_accum_temp, 0) + vget_lane_f32(dot_accum_temp, 1);

#else

    float dot_prod = 0.0f;

    for (int i = 0; i < n_blocks; i++)
    {
        const Q8Block* b0 = inp0 + i;
        const Q4Block* b1 = inp1 + i;

        int block_dot_prod = 0;
        const int half_block_size = block_size / 2;
        for (int j = 0; j < half_block_size; j += 1)
        {
            const Qint4 b00 = b1->data[j];

            const Qint8 b01 = (b00 >> 4) - 7;
            const Qint8 b02 = (b00 & 0b00001111) - 7;

            block_dot_prod += b0->data[j] * b01;
            block_dot_prod += b0->data[j+half_block_size] * b02;
        }

        const float b0_delta = fp16_to_fp32(b0->delta);
        const float b1_delta = fp16_to_fp32(b1->delta);
        dot_prod += block_dot_prod * b0_delta * b1_delta;
    }
#endif

    return dot_prod;
}


float vec_dot_product(const char* inp0, Dtype inp0_dtype, const char* inp1, Dtype inp1_dtype, int vecsize)
{
    switch (inp0_dtype)
    {
        case kQint8: {
            if (inp1_dtype == kQint4) {
                const Q8Block* inp0_data = reinterpret_cast<const Q8Block*>(inp0);
                const Q4Block* inp1_data = reinterpret_cast<const Q4Block*>(inp1);
                return vec_dot_product_q8_q4(inp0_data, inp1_data, vecsize);
            } else {
                const Q8Block* inp0_data = reinterpret_cast<const Q8Block*>(inp0);
                const Q8Block* inp1_data = reinterpret_cast<const Q8Block*>(inp1);
                return vec_dot_product_q8(inp0_data, inp1_data, vecsize);
            }
        }
        case kFloat16: {
            const Float16* inp0_data = reinterpret_cast<const Float16*>(inp0);
            const Float16* inp1_data = reinterpret_cast<const Float16*>(inp1);
            return vec_dot_product_f16(inp0_data, inp1_data, vecsize);
        }
        case kFloat32: {
            const float* inp0_data = reinterpret_cast<const float*>(inp0);
            const float* inp1_data = reinterpret_cast<const float*>(inp1);
            return vec_dot_product_f32(inp0_data, inp1_data, vecsize);
        }
        default: {
            GTEN_ASSERT(false);
            return 0.0f;
        }
    }
}

static void copy_row(const Tensor& src, Tensor& dest, const int src_row_idx, const int dest_row_idx)
{
    const char* src_data = src.data_ptr<char>() + src_row_idx * src.bstride(0);
    char* dest_data = dest.data_ptr<char>() + dest_row_idx * dest.bstride(0);

    if (src.dtype() == kQint8) {
        const size_t copy_nbytes = src.dimsize(1) / globs::q8_block_size * sizeof(Q8Block);
        std::memcpy(dest_data, src_data, copy_nbytes);
    } else if (src.dtype() == kQint4) {
        GTEN_ASSERT(dest.dtype() == kQint8);
        const int rowsize = src.dimsize(1);
        float* inbuf = g_ops_state.buf(rowsize);
        read_row_to_float(src_data, kQint4, inbuf, rowsize);
        write_row_from_float(inbuf, dest_data, kQint8, rowsize);
    }
    else {
        const size_t copy_nbytes = src.dimsize(1) * src.itemsize();
        std::memcpy(dest_data, src_data, copy_nbytes);
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
    // GTEN_ASSERT(weight.dtype() == out.dtype());

    tensor_row_index_impl(weight, tokens, out, start_pos);
}


// void matmul_2d_impl(const Tensor& inp, const Tensor& w, Tensor& out, const int start_pos)
// {
//     const char* inp_data = inp.data_ptr<char>();
//     const char* w_data = w.data_ptr<char>(); 
//     char* out_data = out.data_ptr<char>();

//     const Dtype inp_dtype = inp.dtype();
//     const Dtype w_dtype = w.dtype();
//     const Dtype out_dtype = out.dtype();

//     const int n_ctx = inp.dimsize(0);
//     const int d_inp = inp.dimsize(1);
//     const int d_out = w.dimsize(0);
//     const int inp_st0 = inp.bstride(0);
//     const int w_st0 = w.bstride(0);
//     const int out_st0 = out.bstride(0); 

//     const int block_size = globs::q8_block_size;
//     const int n_blocks = d_out/block_size;

//     int out_c0_stride;
//     if (out_dtype == kQint8) { out_c0_stride = sizeof(Q8Block); }
//     else if (out_dtype == kFloat16) { out_c0_stride = block_size*sizeof(Float16); }
//     else if (out_dtype == kFloat32) { out_c0_stride = block_size*sizeof(float); }
//     else { GTEN_ASSERT(false); }

//     #pragma omp parallel for collapse(2)
//     for (int r0 = start_pos; r0 < n_ctx; r0++) {
//         for (int c0 = 0; c0 < n_blocks; c0++)
//         {
//             const char* inp_row_data = inp_data + r0*inp_st0;

//             float out_buf[block_size];
//             for (int c00 = 0; c00 < block_size; ++c00)
//             {
//                 const char* w_row_data = w_data + (c0 * block_size + c00) * w_st0;
//                 const float dot_prod = vec_dot_product(inp_row_data, inp_dtype, w_row_data, w_dtype, d_inp);
//                 out_buf[c00] = dot_prod;
//             }

//             char* out_row_data = out_data + r0*out_st0 + c0*out_c0_stride;
//             write_row_from_float(out_buf, out_row_data, out_dtype, block_size);
//         }
//     }
// }

static void matmul_2d_impl(const Tensor& inp, const Tensor& w, Tensor& out, const int start_pos)
{
    const char* inp_data = inp.data_ptr<char>();
    const char* w_data = w.data_ptr<char>(); 
    char* out_data = out.data_ptr<char>();

    const Dtype inp_dtype = inp.dtype();
    const Dtype w_dtype = w.dtype();
    const Dtype out_dtype = out.dtype();

    const int n_ctx = inp.dimsize(0);
    const int n_embd = inp.dimsize(1);
    const int d_out = w.dimsize(0);
    const int inp_st0 = inp.bstride(0);
    const int w_st0 = w.bstride(0);
    const int out_st0 = out.bstride(0); 

    float* out_buf = g_ops_state.buf(d_out);

    for (int r0 = start_pos; r0 < n_ctx; r0++) {
        const char* inp_row_data = inp_data + r0*inp_st0;

#if defined(_OPENMP)
        #pragma omp parallel for
#endif
        for (int c0 = 0; c0 < d_out; c0++)
        {
            const char* w_row_data = w_data + c0*w_st0;
            const float dot_prod = vec_dot_product(inp_row_data, inp_dtype, w_row_data, w_dtype, n_embd);
            out_buf[c0] = dot_prod;
        }
        
        char* out_row_data = out_data + r0*out_st0;
        write_row_from_float(out_buf, out_row_data, out_dtype, d_out);
    }
}


static void matmul_2d(const Tensor& x, const Tensor& w, Tensor& out, const int start_pos=0)
{
    const int n_ctx = x.dimsize(0);
    const int n_out = w.dimsize(0);
    const int n_embd = x.dimsize(1);

    GTEN_ASSERT(x.is_2d());
    GTEN_ASSERT(w.is_2d() && w.dimsize(1) == n_embd);
    // GTEN_ASSERT(x.dtype() == w.dtype());
    if (out.is_1d()) {
        GTEN_ASSERT(n_ctx - start_pos == 1);
        GTEN_ASSERT(out.shape_eq({n_out}));
    } else if (out.is_2d()) {
        GTEN_ASSERT(out.shape_eq({n_ctx, n_out}));
    } else {
        GTEN_ASSERT(false);
    }

    matmul_2d_impl(x, w, out, start_pos);
}


static void silu_impl(const Tensor& inp, Tensor& out, const int start_pos)
{
    const char* inp_data = inp.data_ptr<char>();
    const Dtype inp_dtype = inp.dtype();
    char* out_data = out.data_ptr<char>();
    const Dtype out_dtype = out.dtype();

    const int n_ctx = inp.dimsize(0);
    const int n_embd = inp.dimsize(1);
    const int inp_st0 = inp.bstride(0);
    const int out_st0 = out.bstride(0);

    float* out_buf = g_ops_state.buf(n_embd);

    for (int i = start_pos; i < n_ctx; i++) {
        read_row_to_float(inp_data + i * inp_st0, inp_dtype, out_buf, n_embd);

        for (int j = 0; j < n_embd; j++) {
            const float x = out_buf[j];
            out_buf[j] = x / (1.0f + std::exp(-x));
        }

        write_row_from_float(out_buf, out_data + i * out_st0, out_dtype, n_embd);
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
    char* inp_data = inp.data_ptr<char>();
    const Dtype inp_dtype = inp.dtype();

    const int n_ctx = inp.dimsize(0);
    const int n_embd = inp.dimsize(1);
    const int n_head = n_embd / d_head;
    const int inp_st0 = inp.bstride(0);

    Tensor inpx = inp.view({n_ctx, n_head, d_head});

    float* inp_buf = g_ops_state.buf(n_embd);

    const float d = static_cast<float>(d_head);
    for (int i = start_pos; i < n_ctx; ++i) {
        char* inp_row_data = inp_data + i * inp_st0;
        read_row_to_float(inp_row_data, inp_dtype, inp_buf, n_embd);

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

        write_row_from_float(inp_buf, inp_row_data, inp_dtype, n_embd);
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
    const char* inp_data = inp.data_ptr<char>();
    const Dtype inp_dtype = inp.dtype();
    char* out_data = out.data_ptr<char>();
    const Dtype out_dtype = out.dtype();

    const int n_ctx = inp.dimsize(0);
    const int n_embd = inp.dimsize(1);
    const int inp_st0 = inp.bstride(0);
    const int out_st0 = out.bstride(0);

    const Float16* weight_data = weight.data_ptr<Float16>();
    float* inp_buf = g_ops_state.buf(n_embd * 2);
    float* out_buf = inp_buf + n_embd;

    for (int i = start_pos; i < n_ctx; i++) {
        const char* inp_row_data = inp_data + i * inp_st0;
        read_row_to_float(inp_row_data, inp_dtype, inp_buf, n_embd);

        rms_norm_vec_f32(inp_buf, weight_data, out_buf, n_embd);

        write_row_from_float(out_buf, out_data + i * out_st0, out_dtype, n_embd);
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

static void mul_impl(const Tensor& inp0, const Tensor& inp1, Tensor& out, const int start_pos)
{
    const char* inp0_data = inp0.data_ptr<char>();
    const Dtype inp0_dtype = inp0.dtype();
    const char* inp1_data = inp1.data_ptr<char>();
    const Dtype inp1_dtype = inp1.dtype();
    char* out_data = out.data_ptr<char>();
    const Dtype out_dtype = out.dtype();

    const int n_ctx = inp0.dimsize(0);
    const int n_embd = inp0.dimsize(1);
    const int inp0_st0 = inp0.bstride(0);
    const int inp1_st0 = inp1.bstride(0);
    const int out_st0 = out.bstride(0);

    float* x0_buf = g_ops_state.buf(n_embd * 3);
    float* x1_buf = x0_buf + n_embd;
    float* out_buf = x1_buf + n_embd;

    for (int i = start_pos; i < n_ctx; i++)
    {
        read_row_to_float(inp0_data + i * inp0_st0, inp0_dtype, x0_buf, n_embd);
        read_row_to_float(inp1_data + i * inp1_st0, inp1_dtype, x1_buf, n_embd);

        vec_mul_f32(x0_buf, x1_buf, out_buf, n_embd);

        write_row_from_float(out_buf, out_data + i * out_st0, out_dtype, n_embd);
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


static void add_impl(const Tensor& inp0, const Tensor& inp1, Tensor& out, const int start_pos)
{
    const char* inp0_data = inp0.data_ptr<char>();
    const Dtype inp0_dtype = inp0.dtype();
    const char* inp1_data = inp1.data_ptr<char>();
    const Dtype inp1_dtype = inp1.dtype();
    char* out_data = out.data_ptr<char>();
    const Dtype out_dtype = out.dtype();

    // tensor.row_to_float
    const int n_ctx = inp0.dimsize(0);
    const int n_embd = inp0.dimsize(1);
    const int inp0_st0 = inp0.bstride(0);
    const int inp1_st0 = inp1.bstride(0);
    const int out_st0 = out.bstride(0);

    float* x0_buf = g_ops_state.buf(n_embd * 3);
    float* x1_buf = x0_buf + n_embd;
    float* out_buf = x1_buf + n_embd;

    for (int i = start_pos; i < n_ctx; i++) {
        read_row_to_float(inp0_data + i * inp0_st0, inp0_dtype, x0_buf, n_embd);
        read_row_to_float(inp1_data + i * inp1_st0, inp1_dtype, x1_buf, n_embd);

        vec_add_f32(x0_buf, x1_buf, out_buf, n_embd);

        write_row_from_float(out_buf, out_data + i * out_st0, out_dtype, n_embd);
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

/*

All Quantised tensors.

A set of blocks.

d1: h, rows
d2: rows
d3: row

h, n_ctx, n_ctx

n_ctx: blocks.

how many blocks.

*/

void qk_masked_softmax(const Tensor& q, const Tensor& k, Tensor& qk_out, float scale_factor, const int start_pos)
{
    const char* q_data = q.data_ptr<char>();
    const char* k_data = k.data_ptr<char>();
    char* out_data = qk_out.data_ptr<char>();

    const int q_heads = q.dimsize(0);
    const int n_ctx = q.dimsize(1);
    const int d_head = q.dimsize(2);
    const int k_heads = k.dimsize(0);
    const int q_heads_per_group = q_heads / k_heads;

    const int qst0 = q.bstride(0);
    const int qst1 = q.bstride(1);
    const int kst0 = k.bstride(0);
    const int kst1 = k.bstride(1);
    const int qkst0 = qk_out.stride(0);
    const int qkst1 = qk_out.stride(1);

    const Dtype inp_dtype = q.dtype();
    const Dtype out_dtype = qk_out.dtype();

    float* out_buf = g_ops_state.buf(n_ctx);

    for (int qrow = start_pos; qrow < n_ctx; qrow++) {
       for (int h = 0; h < q_heads; h++) {
            // `kcol_max` represents number of the dot products that are not subsequently masked.
            const int kcol_max = qrow + 1;
            for (int kcol = 0; kcol < kcol_max; kcol++) {
                const char* qrow_data = q_data + (h * qst0 + qrow * qst1);
                const char* kcol_data = k_data + ((h / q_heads_per_group) * kst0 + kcol * kst1); // col_data is contigous.

                const float dot_prod = vec_dot_product(qrow_data, inp_dtype, kcol_data, inp_dtype, d_head);
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

            char* out_row_data = out_data + h * qkst0 + qrow * qkst1;
            write_row_from_float(out_buf, out_row_data, out_dtype, n_ctx);
        }
    }
}

// Transpose and dequantize v.
void transpose_v(const Tensor& inp, float* out_buf)
{
    // v.view({n_ctx, kv_n_head, d_head}).permute({1, 2, 0});
    const char* inp_data = inp.data_ptr<char>();
    const int n_head = inp.dimsize(0);
    const int d_head = inp.dimsize(1);
    const int n_ctx = inp.dimsize(2);
    const int n_embd = n_head*d_head;

    // Dequantize v and transpose it.
    const int n_blocks = (n_head*d_head) / globs::q8_block_size;

    switch (inp.dtype())
    {
        case kQint8:
        {
            for (int i = 0; i < n_ctx; i++) {
            for (int j = 0; j < n_blocks; j++) {
                const Q8Block* blk = (Q8Block*)inp_data + i * n_blocks + j;
                const float block_delta = fp16_to_fp32(blk->delta);
                
                for (int k = 0; k < globs::q8_block_size; k++) {
                    const int col_idx = j * globs::q8_block_size + k;
                    out_buf[i + col_idx * n_ctx] = q8_dequantize_single(blk->data[k], block_delta);
                }
            }   
        }
        } break;

        case kFloat16:
        {
            for (int i = 0; i < n_ctx; i++) {
            for (int j = 0; j < n_embd; j++) {
                out_buf[i + j * n_ctx] = fp16_to_fp32(((Float16*)inp_data)[i * n_embd + j]);
            }
        }
        } break;
        default: {
            GTEN_ASSERT(false);
        } break;
    }
}

void qkv_matmul(const Tensor& qk, const Tensor& v, Tensor& qkv_out, const int start_pos)
{
    const char* qk_data = qk.data_ptr<char>();
    char* out_data = qkv_out.data_ptr<char>();

    const int n_ctx = qk.dimsize(1);
    const int q_heads = qk.dimsize(0);
    const int v_heads = v.dimsize(0);
    const int dhead = v.dimsize(1);
    const int q_heads_per_group = q_heads / v_heads;

    const int qkst0 = qk.stride(0);
    const int qkst1 = qk.stride(1);
    const int qkv_st0 = qkv_out.bstride(0);

    const int v_n_embd = v_heads*dhead;
    float* qk_row_buf = g_ops_state.buf(n_ctx + n_ctx*v_n_embd + q_heads*dhead);
    float* v_buf = qk_row_buf + n_ctx;
    float* out_buf = v_buf + n_ctx*v_n_embd;

    // transpose and dequantize v.
    /// TODO: Improve this.
    transpose_v(v, v_buf);

    const Dtype inp_dtype = qk.dtype();
    const Dtype out_dtype = qkv_out.dtype();

    for (int qkr = start_pos; qkr < n_ctx; qkr++) {
        for (int h = 0; h < q_heads; h++) {
            const char* qkr_data = qk_data + (h * qkst0 + qkr * qkst1);  // qk_row_data
            read_row_to_float(qkr_data, inp_dtype, qk_row_buf, n_ctx);

            for (int vc = 0; vc < dhead; vc++) {
                const float* v_col_buf = v_buf + ((h / q_heads_per_group) * dhead*n_ctx + vc * n_ctx);

                const float dot_prod = vec_dot_product_f32(qk_row_buf, v_col_buf, n_ctx);
                out_buf[h*dhead + vc] = dot_prod;
            }
        }

        write_row_from_float(out_buf, out_data + qkr * qkv_st0, out_dtype, q_heads*dhead);      
    }

}


// transposed_out
// 12(f32) 6(f16) 3(8-bit) 1.5(4-bit)

static void qkv_attn_impl(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& qk, Tensor& qkv, int max_ctx, const int start_pos)
{
    const int n_ctx = q.dimsize(0);
    const int n_embd = q.dimsize(1);
    const int q_n_head = qk.dimsize(0);
    const int d_head = n_embd / q_n_head;
    const int kv_n_head = k.dimsize(1) / d_head;

    const Tensor q0 = q.view({n_ctx, q_n_head, d_head}).permute({1, 0, 2});
    
    const Tensor k0 = k.view({n_ctx, kv_n_head, d_head}).permute({1, 0, 2});
    qk.set_strides({max_ctx * max_ctx, max_ctx, 1});

    const float scale_factor = 1.0f / std::sqrt((float)d_head);
    ops::qk_masked_softmax(q0, k0, qk, scale_factor, start_pos);

    Tensor v0 = v.view({n_ctx, kv_n_head, d_head}).permute({1, 2, 0});

    Tensor qkv0 = qkv.view({n_ctx, q_n_head, d_head});
    
    ops::qkv_matmul(qk, v0, qkv0, start_pos);
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
