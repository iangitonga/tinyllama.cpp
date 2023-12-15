#pragma once

#include <chrono>
#include <iostream>

#include "tensor.h"

namespace gten {

/// Provides an embedding table lookup for tokens.
class Embedding {
public:
    Embedding() = default;
    Embedding(int n_vocab, int d_embed, int max_ctx, Dtype dtype);

    /// Returns the embeddings of the given tokens. The input tensor must be of shape
    /// (n_ctx,) and the output tensor is of shape (n_ctx, d_embed).
    Tensor forward(const Tensor& tokens, const int start_pos = 0);

public:
    Tensor weight;
    Tensor emb_acv;
    int64_t exec_time{0};
};


class RMSNorm {
public:
    RMSNorm(int d_in, int max_ctx, Dtype dtype);
    Tensor forward(const Tensor& inp, const int start_pos = 0);

public:
    Tensor weight;
    Tensor acv;
    int64_t exec_time{0};
};

class Residual {
public:
    Residual() = default;
    Residual(int max_ctx, int d_out, Dtype dtype);
    Tensor forward(const Tensor& inp0, const Tensor& inp1, const int start_pos = 0);

public:
    Tensor acv;
    int64_t exec_time{0};
};


/// Applies an affine linear transformation on the input.
class Linear {
public:
    Linear() = default;
    Linear(int d_in, int d_out, int max_ctx, Dtype dtype);
    Tensor forward(const Tensor& inp, const int start_pos = 0);
    Tensor forward_transposed(const Tensor& inp, const int start_pos = 0);

public:
    Tensor weight;
    Tensor acv;
    int64_t exec_time{0};

private:
    int max_ctx_;
    bool has_bias_;
};

class EmbeddingLinear {
public:
    EmbeddingLinear() = default;
    EmbeddingLinear(int n_embd, int n_vocab, int max_ctx, Dtype dtype);
    Tensor forward(const Tensor& inp);

public:
    Tensor weight;
    Tensor acv;
    int64_t exec_time{0};
};

class Multiply {
public:
    Multiply() = default;
    Multiply(int max_ctx, int d_out, Dtype dtype, const bool inplace = false);
    Tensor forward(Tensor& inp0, const Tensor& inp1, const int start_pos=0);

public:
    Tensor acv;
    int64_t exec_time{0};

private:
    bool inplace_{false};
};

class SiLU {
public:
    SiLU() = default;
    SiLU(int max_ctx, int d_out, Dtype dtype, const bool inplace=false);
    Tensor forward(Tensor& inp, const int start_pos=0);

public:
    Tensor acv;
    bool inplace_{false};
    int64_t exec_time{0};
};


class RotaryEmbedding {
public:
    RotaryEmbedding(const int d_head, const bool inplace=true);
    Tensor forward(Tensor& inp, const int start_pos=0);

public:
    int64_t exec_time{0};

private:
    int d_head_;
};


class SelfAttention {
public:
    SelfAttention(int n_heads, int n_embed, int n_query_groups, int max_ctx, Dtype dtype);
    Tensor forward(const Tensor& inp, const int start_pos);

public:
    Linear query;
    Linear key;
    Linear value;
    Linear qkv_proj;
    Tensor qk_acv;
    Tensor qkv_acv;
    RotaryEmbedding q_rope;
    RotaryEmbedding k_rope;
    int64_t exec_time_attn{0};

private:
    int32_t n_heads_;
    int max_ctx_;

private:
    Tensor masked_qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v, const int start_pos);
};


class AttentionBlock {
public:
    AttentionBlock(int n_heads, int d_embed, int n_query_groups, int n_mlp, int max_ctx, Dtype dtype);
    Tensor forward(Tensor& inp, const int start_pos);
    Tensor ffn_forward(const Tensor& inp, const int start_pos=0);

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
