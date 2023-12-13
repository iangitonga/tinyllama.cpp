#include "gten/gten.h"

#include "tokenizer.h"

#include <string_view>

using namespace gten;


struct TinyLLamaParams {
    const int n_vocab = 32003;
    const int max_ctx = 2048;
    const int n_embd = 2048;
    const int n_ffn = 5632;
    const int n_layers = 22;
    const int n_heads = 32;
    const int n_query_groups = 4;
};


class TinyLlama {
public:
    const TinyLLamaParams params = TinyLLamaParams{};
    Dtype dtype_;

public:
    TinyLlama(Dtype dtype, int qblock_size=64)
        : dtype_{dtype},
          tok_emb_{Embedding(params.n_vocab, params.n_embd, params.max_ctx, dtype, qblock_size)},
          norm_{RMSNorm(params.n_embd, params.max_ctx, dtype, qblock_size)},
          lm_head_{EmbeddingLinear{params.n_embd, params.n_vocab, params.max_ctx, dtype, qblock_size}}
    {
        blocks_.reserve(params.n_layers);
        for (int i = 0; i < params.n_layers; i++) {
            blocks_.push_back(AttentionBlock(params.n_heads, params.n_embd, params.n_query_groups, params.n_ffn, params.max_ctx, dtype, qblock_size));
        }
    }

    Tensor logits(const Tensor& tokens) {
        Tensor logits = tok_emb_.forward(tokens);

        for (auto& block : blocks_) {
            logits = block.forward(logits);
        }

        logits = norm_.forward(logits);
        logits = lm_head_.forward(logits);

        return logits;
    }

    void load_from_ckpt(std::ifstream& ckpt);

private:
    Embedding tok_emb_;
    RMSNorm norm_;
    EmbeddingLinear lm_head_;
    std::vector<AttentionBlock> blocks_;
};


/*

prompt format: <|im_start|>user\nPROMPT<|im_end|>\n<|im_start|>assistant\n

---------------------------+
TOKEN         | TOKEN_ID   |
---------------------------+
bos_tok       | 1          |
<|im_start|>  | 32001      |
user          | 1404       |
\n            | 13         |
" \n"         | 29871, 13  |
<|im_end|>    | 32002      |
assistance    | 20255      |
---------------------------+

Example prompt: Who is Karl Marx?
tokens: 1, 32001, 1404, 13, 22110, 338, 8425, 28579, 29973, 32002, 29871, 13, 32001, 20255, 13
output: 24115, 29880, 28579, 338, 263, 5332, 8578, 359, 13434, 322, 7766, 391, 1058, 338, 5545,
        697, 310, 278, 1556, 4100, 13994, 297, 278, 5849, 310, 28579, 391, 6368, 322, 6944

*/
/*

./tinyllama -p PROMPT
./tinyllama -q8 -p PROMPT

*/

static const char* usage_message = "USAGE:\n ./tinyllama [-q8] -p PROMPT\n option: -q8 for the quantized version of the model.\n";


int main(int argc, char const *argv[])
{
    if (argc < 3) {
        std::cerr << "Incorrect number of arguments.\n";
        std::cerr << usage_message;
        std::exit(EXIT_FAILURE);
    }

    Dtype model_dtype = kFloat16;
    std::string model_path = "models/tinyllama.fp16.gten";
    std::string prompt = "";
    for (int i = 1; i < argc; i++)
    {
        std::string_view arg{argv[i]};
        if (arg == "-q8") {
            model_dtype = kQint8;
            model_path = "models/tinyllama.q8.gten";
        } else if (arg == "-p") {
            if (i + 1 < argc) {
                prompt = argv[i + 1];
                i += 1; // fast-forward
            } else {
                std::cerr << "error: Prompt not provided.\n" << usage_message << "\n";
                std::exit(EXIT_FAILURE);
            }
        } else {
            std::cerr << "error: Unknown argument: " << arg << "\n" << usage_message;
            std::exit(EXIT_FAILURE);
        }
    }

    if (prompt == "") {
        std::cerr << "error: Prompt not provided.\n" << usage_message << "\n";
        std::exit(EXIT_FAILURE);
    }
    
    std::string model_id = model_dtype == kFloat16 ? "fp16" : "q8";

#ifdef _WIN32
    int res = std::system(("python model_dl.py " + model_id).c_str());
#else
    int res = std::system(("python3 model_dl.py " + model_id).c_str());
#endif
    if (res != 0) {
        std::cerr << "Error: Failed to download the model. Check your network connectivity.\n";
        return -1;
    }

    std::ifstream checkpoint{model_path, std::ios::binary};
    GTEN_ASSERT(checkpoint.is_open());

    TinyLlama model{model_dtype};
    model.load_from_ckpt(checkpoint);

    Tokenizer tokenizer{"tokenizer.bin", 32000};
    std::vector<int> toks = tokenizer.encode(prompt);
    toks.reserve(model.params.max_ctx);

    const int niter = model.params.max_ctx - toks.size();
    for (int i = 0; i < niter; i++)
    {
        Tensor tokens{toks.data(), {(int)toks.size()}, kInt32};

        Tensor logits = model.logits(tokens);

        const int n_ctx = logits.size(0);
        const float *logits_data = logits.data_ptr<float>();
        const int logits_size = model.params.n_vocab;

        float max_prob = -std::numeric_limits<float>::infinity();
        int max_index = 0;
        for (int j = 0; j < logits_size; ++j){
            const float val = logits_data[j];
            if (val > max_prob) {
                max_prob = val;
                max_index = j;
            }
        }

        const int maxi = max_index;
        if (maxi == tokenizer.eos) {
            break;
        }
        const int prev_token = (i == 0) ? 1 : toks.back();
        std::cerr << tokenizer.decode(prev_token, maxi);

        toks.push_back(maxi);
    }
    
    std::cerr << '\n';

    return 0;
}


static inline void read_into_weight(
    std::ifstream& fin, gten::Tensor& tensor, Dtype dtype)
{
    std::string weight_name;
    int32_t weight_name_size;
    fin.read(reinterpret_cast<char*>(&weight_name_size), sizeof(weight_name_size));
    weight_name.resize(weight_name_size);
    fin.read(reinterpret_cast<char*>(weight_name.data()), weight_name_size);

    if (dtype == kQint8)
    {
        GTEN_ASSERT(tensor.dtype() == kQint8);
        
        int32_t deltas_bytes;
        fin.read(reinterpret_cast<char*>(&deltas_bytes), sizeof(deltas_bytes));
        const int ndeltas = deltas_bytes / sizeof(Float16);

        Qparams qparams = tensor.qparams();
        const int expected_deltas = qparams.n_deltas();
        GTEN_ASSERTM(ndeltas == expected_deltas, "expected %d but got %d deltas.", expected_deltas, ndeltas);

        Float16* deltas = qparams.deltas();
        fin.read(reinterpret_cast<char*>(deltas), deltas_bytes); /// deltas size.
    }

    int32_t weight_payload_size;
    fin.read(reinterpret_cast<char*>(&weight_payload_size), sizeof(weight_payload_size));

    // if (debug)
        // std::cout << weight_name << " (" << weight_payload_size << ")\n";

    GTEN_ASSERTM(
        static_cast<size_t>(weight_payload_size) == tensor.nbytes(),
        "Weight `%s` data size: %d does not match the expected size: %ld.",
        weight_name.c_str(), weight_payload_size, tensor.nbytes());
    fin.read(tensor.data_ptr<char>(), weight_payload_size);
}


static inline void read_layer_header(std::ifstream& fin, bool debug = false) {
    std::string layer_name;
    int32_t layer_name_size;
    fin.read(reinterpret_cast<char*>(&layer_name_size), sizeof(layer_name_size));
    layer_name.resize(layer_name_size);
    fin.read(reinterpret_cast<char*>(layer_name.data()), layer_name_size);

    if (debug) {
        std::cout << "Layer: " << layer_name << "\n";
    }
}

void TinyLlama::load_from_ckpt(std::ifstream &ckpt)
{
    const int64_t expected_magic = 0x454c49464e455447;
    int64_t magic;
    ckpt.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    GTEN_ASSERTM(magic == expected_magic, "Magic number in the binary does not match the expected one.\n");

    read_layer_header(ckpt);
    read_into_weight(ckpt, tok_emb_.weight, dtype_);

    for (auto& block : blocks_)
    {
        // q_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.query.weight, dtype_);

        // k_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.key.weight, dtype_);

        // v_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.value.weight, dtype_);

        // o_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.qkv_proj.weight, dtype_);

        // ffn_gate_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_gate_proj.weight, dtype_);

        // ffn_up_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_up_proj.weight, dtype_);

        // ffn_down_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_down_proj.weight, dtype_);

        // attn_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn_norm.weight, kFloat16);

        // ffn_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_norm.weight, kFloat16);
    }
    
    read_layer_header(ckpt);
    read_into_weight(ckpt, norm_.weight, kFloat16);

    read_layer_header(ckpt);
    read_into_weight(ckpt, lm_head_.weight, dtype_);
}
