#include <cmath>
#include <iostream>
#include <cstring>

#include "modules.h"
#include "ops.h"


namespace gten {

Embedding::Embedding(int n_vocab, int n_embd, int max_ctx, Dtype dtype, int qblock_size)
    : weight{Tensor({n_vocab, n_embd}, dtype, qblock_size)},
      emb_acv{Tensor({max_ctx, n_embd}, dtype, qblock_size)},
      max_ctx_(max_ctx)
{
}

Tensor Embedding::forward(const Tensor& tokens) {
    Timer timer{&exec_time_emb_ms_};
    
    const int n_embd = weight.size(1);
    emb_acv.resize({tokens.numel(), n_embd});

    if (emb_acv_cached_) {
        ops::token_embed(weight, tokens, emb_acv, /*last_token_only=*/true);
    } else {
        emb_acv_cached_ = true;

        ops::token_embed(weight, tokens, emb_acv);
    }

    return emb_acv;
}

Residual::Residual(int max_ctx, int n_out, Dtype dtype, int qblock_size)
    : acv{Tensor({max_ctx, n_out}, dtype, qblock_size)}, max_ctx_{max_ctx}
{
}

Tensor Residual::forward(const Tensor& inp0, const Tensor& inp1) {
    Timer timer{&exec_time_ms_};

    const int n_ctx = inp0.size(0);
    const int n_embd = inp0.size(1);

    acv.resize({n_ctx, n_embd});

    if (acv_cached_) {
        ops::add(inp0, inp1, acv, /*last_ctx_only=*/true);
    } else {
        acv_cached_ = true;
        ops::add(inp0, inp1, acv);
    }

    return acv;
}

Linear::Linear(int n_in, int n_out, int max_ctx, Dtype dtype, int qblock_size)
    : weight{Tensor({n_out, n_in}, dtype, qblock_size)},
      acv{Tensor({max_ctx, n_out}, dtype, qblock_size)},
      max_ctx_{max_ctx}
{
}

Tensor Linear::forward(const Tensor &inp) {
    Timer timer{&exec_time_ms_};

    const int n_ctx = inp.size(0);
    const int n_out = weight.size(0);
    
    acv.resize({n_ctx, n_out});

    if (acv_cached_) {
        ops::matmul_2d(inp, weight, acv, /*last_ctx_only=*/true);
    } else {
        acv_cached_ = true;

        ops::matmul_2d(inp, weight, acv);
    }

    return acv;
}


/// TODO: Construct a transposed linear module?
Tensor Linear::forward_transposed(const Tensor &inp) {
    Timer timer{&exec_time_ms_};

    const int n_ctx = inp.size(0);
    const int n_out = weight.size(0);
    
    acv.resize({n_out, n_ctx});
    /// TODO: Allow strides-lock on tensors.
    acv.set_strides({max_ctx_, 1});

    if (acv_cached_) {
        ops::matmul_2d_transposed(inp, weight, acv, /*last_ctx_only=*/true);
    } else {
        acv_cached_ = true;
        ops::matmul_2d_transposed(inp, weight, acv);
    }

    return acv;
}

EmbeddingLinear::EmbeddingLinear(int n_embd, int n_vocab, int max_ctx, Dtype dtype, int qblock_size)
    : weight{Tensor({n_vocab, n_embd}, dtype, qblock_size)}, acv{Tensor({n_vocab}, kFloat32)}
{
}

Tensor EmbeddingLinear::forward(const Tensor& inp)
{
    ops::emb_matmul(inp, weight, acv);
    return acv;
}

RMSNorm::RMSNorm(int d_in, int max_ctx, Dtype dtype, int qblock_size)
    : weight{Tensor({d_in}, kFloat16)}, acv{Tensor({max_ctx, d_in}, dtype, qblock_size)}
{
}

Tensor RMSNorm::forward(const Tensor& inp)
{
    const int n_ctx = inp.size(0);
    const int n_embd = inp.size(1);

    acv.resize({n_ctx, n_embd});

    if (acv_cached_) {
        ops::rms_norm(inp, weight, acv, n_ctx - 1);
    } else {
        acv_cached_ = true;
        ops::rms_norm(inp, weight, acv);
    }

    return acv;
}

Multiply::Multiply(int max_ctx, int d_out, Dtype dtype, int qblock_size)
    : acv{Tensor({max_ctx, d_out}, dtype, qblock_size)}
{
}

Tensor Multiply::forward(const Tensor &inp0, const Tensor &inp1)
{
    const int n_ctx = inp0.size(0);
    const int n_embd = inp0.size(1);

    acv.resize({n_ctx, n_embd});

    ops::mul(inp0, inp1, acv);
    return acv;
}

SiLU::SiLU(int max_ctx, int d_out, Dtype dtype, int qblock_size)
    : acv{Tensor({max_ctx, d_out}, dtype, qblock_size)}
    {
    }

Tensor SiLU::forward(const Tensor &inp)
{
    const int n_ctx = inp.size(0);
    const int n_embd = inp.size(1);

    acv.resize({n_ctx, n_embd});

    ops::silu(inp, acv);
    return acv;
}

/// TODO: Change size to dimsize[i] or shape[i].

Tensor RotaryEmbedding::forward(Tensor& inp)
{
    const int n_ctx = inp.size(0);
    if (acv_cached_) {
        ops::rotary_emb(inp, /*start_pos=*/n_ctx-1);
    } else {
        acv_cached_ = true;

        ops::rotary_emb(inp);
    }

    return inp;
}


SelfAttention::SelfAttention(int n_heads, int n_embd, int n_query_groups, int max_ctx, Dtype dtype, int qblock_size)
    : query{Linear(n_embd, n_embd, max_ctx, dtype, qblock_size)},
      qkv_proj{Linear(n_embd, n_embd, max_ctx, dtype, qblock_size)},
      qk_acv{Tensor({n_heads, max_ctx, max_ctx}, dtype, qblock_size, /*zero_mem=*/true)},
      qkv_acv{Tensor({max_ctx, n_embd}, dtype, qblock_size)},
      qrot{RotaryEmbedding{}},
      krot{RotaryEmbedding{}},
      n_heads_{n_heads}, max_ctx_{max_ctx}
{
    const int d_head = n_embd / n_heads;
    const int kv_dim = d_head * n_query_groups;
    key = Linear{n_embd, kv_dim, max_ctx, dtype, qblock_size};
    value = Linear{n_embd, kv_dim, max_ctx, dtype, qblock_size};
}

Tensor SelfAttention::forward(const Tensor &inp)
{
    Tensor q = query.forward(inp);
    Tensor k = key.forward(inp);

    q = qrot.forward(q);
    k = krot.forward(k);

    Tensor v = value.forward_transposed(inp);

    const Tensor qkv = masked_qkv_attn(q, k, v);
    const Tensor out = qkv_proj.forward(qkv);
    return out;
}

Tensor SelfAttention::masked_qkv_attn(const Tensor &q, const Tensor &k, const Tensor &v)
{
    const int n_ctx = q.size(0);
    const int n_embd = q.size(1);

    qk_acv.resize({n_heads_, n_ctx, n_ctx});
    qkv_acv.resize({n_ctx, n_embd});

    ops::qkv_attn(q, k, v, qk_acv, qkv_acv, max_ctx_);

    return qkv_acv;
}

AttentionBlock::AttentionBlock(int n_heads, int n_embd, int n_query_groups, int n_mlp, int max_ctx, Dtype dtype, int qblock_size)
    : attn_norm{RMSNorm(n_embd, max_ctx, dtype, qblock_size)},
      attn{SelfAttention(n_heads, n_embd, n_query_groups, max_ctx, dtype, qblock_size)},
      inp_res{Residual(max_ctx, n_embd, dtype, qblock_size)},
      ffn_norm{RMSNorm(n_embd, max_ctx, dtype, qblock_size)},
      ffn_mul{Multiply(max_ctx, n_mlp, dtype, qblock_size)},
      ffn_gate_proj{Linear(n_embd, n_mlp, max_ctx, dtype, qblock_size)},
      ffn_up_proj{Linear(n_embd, n_mlp, max_ctx, dtype, qblock_size)},
      ffn_down_proj{Linear(n_mlp, n_embd, max_ctx, dtype, qblock_size)},
      attn_res{Residual(max_ctx, n_embd, dtype, qblock_size)},
      ffn_silu({SiLU(max_ctx, n_mlp, dtype, qblock_size)})
{
}

Tensor AttentionBlock::ffn_forward(const Tensor& inp) {
    // self.w2(F.silu(self.w1(x)) * self.w3(x))
    const Tensor w1 = ffn_gate_proj.forward(inp);
    const Tensor w3 = ffn_up_proj.forward(inp);
    const Tensor sw1 = ffn_silu.forward(w1);
    const Tensor w1w2 = ffn_mul.forward(sw1, w3);
    Tensor out = ffn_down_proj.forward(w1w2);
    return out;
}

Tensor AttentionBlock::forward(Tensor &inp)
{
    Tensor h = inp_res.forward(inp, attn.forward(attn_norm.forward(inp)));
    Tensor out = attn_res.forward(h, ffn_forward(ffn_norm.forward(h)));
    return out;
}

} // namespace gten
