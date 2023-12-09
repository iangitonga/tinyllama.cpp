#pragma once

#include <chrono>
#include <iostream>

#include "tensor.h"

namespace gten {

/// Provides an embedding table lookup for tokens.
class Embedding {
public:
    Embedding() = default;
    Embedding(int n_vocab, int d_embed, int max_ctx, Dtype dtype, int qblock_size = 0);

    /// Returns the embeddings of the given tokens. The input tensor must be of shape
    /// (n_ctx,) and the output tensor is of shape (n_ctx, d_embed).
    Tensor forward(const Tensor& tokens);

    int64_t emb_time() const { return exec_time_emb_ms_; }
    void reset_acv_cache() { emb_acv_cached_=false; }

public:
    Tensor weight;
    Tensor emb_acv;

private:
    bool emb_acv_cached_{false};
    int max_ctx_;
    int64_t exec_time_emb_ms_{0};
};


class RMSNorm {
public:
    RMSNorm(int d_in, int max_ctx);
    Tensor forward(const Tensor& inp);

public:
    Tensor weight;
    Tensor acv;
    bool acv_cached_{false};
};

class Residual {
public:
    Residual() = default;
    Residual(int max_ctx, int d_out, Dtype dtype, int qblock_size = 0);
    Tensor forward(const Tensor& inp0, const Tensor& inp1);
    int64_t time() const noexcept { return exec_time_ms_; }
    void reset_acv_cache() { acv_cached_=false; }

public:
    Tensor acv;

private:
    int max_ctx_;
    bool acv_cached_{false};
    int64_t exec_time_ms_{0};
};


/// Applies an affine linear transformation on the input.
class Linear {
public:
    Linear() = default;
    Linear(int d_in, int d_out, int max_ctx, Dtype dtype, int qblock_size = 0);
    Tensor forward(const Tensor& inp);
    Tensor forward_transposed(const Tensor& inp);
    int64_t time() const noexcept { return exec_time_ms_; }
    void reset_acv_cache() { acv_cached_=false; }

public:
    Tensor weight;
    Tensor acv;

private:
    bool acv_cached_{false};
    int64_t exec_time_ms_{0};
    int max_ctx_;
    bool has_bias_;
};

class EmbeddingLinear {
public:
    EmbeddingLinear() = default;
    EmbeddingLinear(int n_embd, int n_vocab, int max_ctx);
    Tensor forward(const Tensor& inp);

public:
    Tensor weight;
    Tensor acv;
};

class Multiply {
public:
    Multiply() = default;
    Multiply(int max_ctx, int d_out);
    Tensor forward(const Tensor& inp0, const Tensor& inp1);

public:
    Tensor acv;
};

class SiLU {
public:
    SiLU() = default;
    SiLU(int max_ctx, int d_out) : acv{Tensor({max_ctx, d_out}, kFloat16)} {}
    Tensor forward(const Tensor& inp);

public:
    Tensor acv;
};


class RotaryEmbedding {
public:
    Tensor forward(Tensor& inp);

private:
    bool acv_cached_{false};
};


class SelfAttention {
public:
    SelfAttention(int n_heads, int n_embed, int n_query_groups, int max_ctx);
    Tensor forward(const Tensor& inp);

public:
    Linear query;
    Linear key;
    Linear value;
    Linear qkv_proj;
    Tensor qk_acv;
    Tensor qkv_acv;
    RotaryEmbedding qrot;
    RotaryEmbedding krot;

private:
    int32_t n_heads_;
    int max_ctx_;

private:
    Tensor masked_qkv_attn(const Tensor &q, const Tensor &k, const Tensor &v);
};


class AttentionBlock {
public:
    AttentionBlock(int n_heads, int d_embed, int n_query_groups, int n_mlp, int max_ctx);
    Tensor forward(Tensor& inp);
    Tensor ffn_forward(const Tensor& inp);

public:
    RMSNorm attn_norm;
    SelfAttention attn;
    Residual inp_res;
    RMSNorm ffn_norm;
    Linear ffn_gate_proj;
    Linear ffn_up_proj;
    Linear ffn_down_proj;
    Residual attn_res;
    Multiply ffn_mul;
    SiLU ffn_silu;
};


class Timer {
public:
    Timer(int64_t* time_tracker)
        : time_tracker_{time_tracker}, start_time_{std::chrono::high_resolution_clock::now()}
    { 
    }
    ~Timer() { stop(); }

    void stop() {
        if (stopped_)
            return;
        auto end_time = std::chrono::high_resolution_clock::now();
        int64_t start = std::chrono::time_point_cast<std::chrono::milliseconds>(start_time_).time_since_epoch().count();
        int64_t end = std::chrono::time_point_cast<std::chrono::milliseconds>(end_time).time_since_epoch().count();
        int64_t duration = end - start;
        *time_tracker_ += duration;
        stopped_ = true;
    }
private:
    int64_t* time_tracker_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    bool stopped_ = false;
};

} // namespace gten
