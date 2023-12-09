#include "gten/gten.h"

#include "tokenizer.h"

#include <string_view>

using namespace gten;


struct TinyLLamaParams {
    const int n_vocab = 32003;
    const int max_ctx = 128;
    const int n_embd = 2048;
    const int n_ffn = 5632;
    const int n_layers = 22;
    const int n_heads = 32;
    const int n_query_groups = 4;
};


class TinyLlama {
public:
    const TinyLLamaParams params = TinyLLamaParams{};

public:
    TinyLlama()
        : tok_emb_{Embedding(params.n_vocab, params.n_embd, params.max_ctx, kFloat16)},
          norm_{RMSNorm(params.n_embd, params.max_ctx)},
          lm_head_{EmbeddingLinear(params.n_embd, params.n_vocab, params.max_ctx)}
    {
        blocks_.reserve(params.n_layers);
        for (int i = 0; i < params.n_layers; i++) {
            blocks_.push_back(AttentionBlock(params.n_heads, params.n_embd, params.n_query_groups, params.n_ffn, params.max_ctx));
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

int main(int argc, char const *argv[])
{
    if (argc < 3) {
        std::cout << "Incorrect number of arguments.\n";
        std::cout << "USAGE: ./tinyllama -p PROMPT\n";
        std::exit(EXIT_FAILURE);
    }

    if (std::string_view(argv[1]) != "-p") {
        std::cout << "Unrecognised option:" << argv[1] <<  "\n";
        std::cout << "USAGE: ./tinyllama -p PROMPT\n";
        std::exit(EXIT_FAILURE);
    }

    std::string prompt = argv[2];

#ifdef _WIN32
    int res = std::system("python model_dl.py");
#else
    int res = std::system("python3 model_dl.py");
#endif
    if (res != 0) {
        std::cerr << "Error: Failed to download the model. Check your network connectivity.\n";
        return -1;
    }

    std::ifstream checkpoint{"models/tinyllama.fp16.gten", std::ios::binary};
    GTEN_ASSERT(checkpoint.is_open());

    TinyLlama model{};
    model.load_from_ckpt(checkpoint);

    Tokenizer tokenizer{"tokenizer.bin", 32000};
    // std::vector<int> toks = {1, 32001, 1404, 13}; 24115, 29880, 28579,
    // std::vector<int> toks = {1, 32001, 1404, 13, 22110, 338, 8425, 28579, 29973, 32002, 29871, 13, 32001, 20255, 13};
    std::vector<int> toks = tokenizer.encode(prompt);
    toks.reserve(model.params.max_ctx);

    for (int i = 0; i < model.params.max_ctx; i++)
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
            // std::cerr << "<EOT>";
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
    std::ifstream& fin, gten::Tensor& tensor, bool debug = false)
{
    std::string weight_name;
    int32_t weight_name_size;
    fin.read(reinterpret_cast<char*>(&weight_name_size), sizeof(weight_name_size));
    weight_name.resize(weight_name_size);
    fin.read(reinterpret_cast<char*>(weight_name.data()), weight_name_size);

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
    read_into_weight(ckpt, tok_emb_.weight);

    for (auto& block : blocks_)
    {
        // q_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.query.weight);

        // k_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.key.weight);

        // v_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.value.weight);

        // o_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.qkv_proj.weight);

        // ffn_gate_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_gate_proj.weight);

        // ffn_up_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_up_proj.weight);

        // ffn_up_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_down_proj.weight);

        // ffn_up_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn_norm.weight);

        // ffn_up_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_norm.weight);
    }
    
    read_layer_header(ckpt);
    read_into_weight(ckpt, norm_.weight);

    read_layer_header(ckpt);
    read_into_weight(ckpt, lm_head_.weight);
}
