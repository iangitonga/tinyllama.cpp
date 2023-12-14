#include "gten/gten.h"

#include "tokenizer.h"

#include <string_view>
#include <random>
#include <algorithm>

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
    int n_ctx_;

public:
    TinyLlama(const int n_ctx, Dtype dtype)
        : n_ctx_{n_ctx},
          dtype_{dtype},
          tok_emb_{Embedding(params.n_vocab, params.n_embd, n_ctx, dtype)},
          norm_{RMSNorm(params.n_embd, n_ctx, dtype)},
          lm_head_{EmbeddingLinear{params.n_embd, params.n_vocab, n_ctx, dtype}}
    {
        blocks_.reserve(params.n_layers);
        for (int i = 0; i < params.n_layers; i++) {
            blocks_.push_back(
                AttentionBlock(params.n_heads, params.n_embd, params.n_query_groups, params.n_ffn, n_ctx, dtype)
            );
        }
    }

    Tensor logits(const Tensor& tokens, const int start_pos=0) {
        if (tokens.numel() > n_ctx_) {
            std::cerr << "Number of prompt tokens (" << tokens.numel() << ") exceed provided maximum ctx size (" << n_ctx_ << ")\n";
            std::exit(EXIT_FAILURE);
        }

        Tensor logits = tok_emb_.forward(tokens, start_pos);

        for (auto& block : blocks_) {
            logits = block.forward(logits, start_pos);
        }

        logits = norm_.forward(logits, start_pos);
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


void greedy_sample(
    std::string& prompt, TinyLlama& model, Tokenizer& tokenizer, const int n_predict);
void topk_sample(
    std::string& prompt, TinyLlama& model, Tokenizer& tokenizer, const int n_predict, const float temp, const int topk);


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

static const char *usage_message = R"(
USAGE:
./tinyllama [options] -p PROMPT  for a single prompt or
./tinyllama [options] for a chat interface. 

Optional args. 
-f16 :     Use float-16 model and inference (2.2GB). [default]
-q8  :     Use 8-bit quantized model (1.1GB) and inference.
--temp T : Temperature to use during sampling. It must be greater than 0. [default=0.9].
--npred  N : Number of tokens to generate. Minimum is 1 and max is 2048. [default=512].
--topk K : Top tokens to randomly select from during prediction. [default=40].

Examples:
  ./tinyllama
  ./tinyllama -q8 --temp 0.5
  ./tinyllama -p "Give three tips for staying healthier." 
)";


int main(int argc, char const *argv[])
{
    Dtype model_dtype = kFloat16;
    std::string model_path = "models/tinyllama.fp16.gten";
    std::string prompt = "";
    int n_predict = 512;
    bool use_greedy_sampler = false;
    float sampling_temp = 0.9f;
    int topk = 40;

    for (int i = 1; i < argc; i++)
    {
        std::string_view arg{argv[i]};
        if (arg == "--help" || arg == "-h") {
            std::cout << usage_message << "\n";
            return 0;
        }
        if (arg == "-f16") {
            continue;
        }
        else if (arg == "-q8") {
            model_dtype = kQint8;
            model_path = "models/tinyllama.q8.gten";
        }
        else if (arg == "-p") {
            if (i + 1 < argc) {
                prompt = argv[i + 1];
                i += 1; // fast-forward
            } else {
                std::cerr << "error: Prompt not provided.\n" << usage_message << "\n";
                std::exit(EXIT_FAILURE);
            }
        } else if (arg == "-greedy") {
           use_greedy_sampler = true;
        } else if (arg == "--npred") {
            if (argc <= i+1) {
                std::cerr << "npred value is missing.\n";
                return -1;
            }
            int npred;
            try {
                npred = std::stoi(argv[i+1]);
            } catch (...) {
                std::cerr << "Invalid npred value.\n";
                return -1;
            }
            if (npred < 1 || npred > 2048) {
                std::cerr << "npred must be greater than 1 and less than 2048.\n";
                return -1;
            }
            n_predict = std::max(npred, 2048);
            i += 1; // skip len param
        } else if (arg == "--temp") {
            if (argc <= i+1) {
                std::cerr << "temp value is missing.\n";
                return -1;
            }
            float arg_temp;
            try {
                arg_temp = std::stof(argv[i+1]);
            } catch (...) {
                std::cerr << "Invalid temp value \n";
                return -1;
            }
            if (arg_temp <= 0.0f) {
                std::cerr << "temp value must be greater than zero.\n";
                return -1;
            }
            sampling_temp = arg_temp;
            i += 1; // skip parsed temp.
        } else if (arg == "--topk") {
            if (argc <= i+1) {
                std::cerr << "topk value is missing.\n";
                return -1;
            }
            int arg_top_k;
            try {
                arg_top_k = std::stoi(argv[i+1]);
            } catch (...) {
                std::cerr << "Invalid topk value.\n";
                return -1;
            }
            const int n_vocab = 32003;
            if (arg_top_k < 1 || arg_top_k > 32003) {
                std::cerr << "topk must be gte 1 and lte " << 32003 << ".\n";
                return -1;
            }
            topk = arg_top_k;
            i += 1;
        }
        else {
            std::cerr << "error: Unknown argument: " << arg << "\n" << usage_message;
            std::exit(EXIT_FAILURE);
        }
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

    TinyLlama model{n_predict, model_dtype};
    model.load_from_ckpt(checkpoint);

    Tokenizer tokenizer{"tokenizer.bin", 32000};
    
    if (prompt == "") {
        std::cout << "Chat interface. Write your prompt and press enter to submit. Enter q or press ctrl+c to quit.\n";
        std::string prompt;
        while (true) {
            std::cerr << "\n\n[You]: ";
            std::getline(std::cin, prompt);
            if (prompt == "q")
                break;

            std::cerr << "\n[Tinyllama-Chat]: \n\n";
            if (use_greedy_sampler) {
                greedy_sample(prompt, model, tokenizer, n_predict);
            } else {
                topk_sample(prompt, model, tokenizer, n_predict, sampling_temp, topk);
            }
        }
    }
    else {
        if (use_greedy_sampler) {
            greedy_sample(prompt, model, tokenizer, n_predict);
        } else {
            topk_sample(prompt, model, tokenizer, n_predict, sampling_temp, topk);
        }
    }

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


void greedy_sample(std::string& prompt, TinyLlama& model, Tokenizer& tokenizer, const int n_predict)
{
    std::vector<int> tokens = tokenizer.encode(prompt);
    tokens.reserve(n_predict);

    const int n_pred_tokens = n_predict - tokens.size();
    for (int i = 0; i < n_pred_tokens; i++)
    {
        Tensor input{tokens.data(), {(int)tokens.size()}, kInt32};

        const int start_pos = (i == 0) ? 0 : input.numel() - 1; 
        Tensor logits = model.logits(input, start_pos);

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
        const int prev_token = (i == 0) ? 1 : tokens.back();
        std::cerr << tokenizer.decode(prev_token, maxi);

        tokens.push_back(maxi);
    }
    
    std::cerr << '\n';
}

void topk_sample(std::string& prompt, TinyLlama& model, Tokenizer& tokenizer, const int n_predict, const float temp, const int top_k)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int> tokens = tokenizer.encode(prompt);
    tokens.reserve(n_predict);
    const int logits_size = model.params.n_vocab;
    std::vector<std::pair<double, int>> logits_probs;
    logits_probs.reserve(logits_size);

    const int eot_token = tokenizer.eos;

    const int n_pred_tokens = n_predict - tokens.size();
    for (int i = 0; i < n_pred_tokens; i++)
    {
        gten::Tensor input{tokens.data(), {(int)tokens.size()}, gten::kInt32};
        const int start_pos = (i == 0) ? 0 : input.numel() - 1; 
        gten::Tensor logits = model.logits(input, start_pos);
        const float* logits_data = logits.data_ptr<float>();

        logits_probs.clear();
        for (int j = 0; j < logits_size; ++j) {
            logits_probs.push_back(std::make_pair((double)logits_data[j] / temp, j));
        }
        
        // Select top k elements.
        std::partial_sort(
                logits_probs.begin(),
                logits_probs.begin() + top_k,
                logits_probs.end(),
                [](const std::pair<double, int> &rhs, const std::pair<double, int> &lhs) {
            return rhs.first > lhs.first;
        });
        logits_probs.resize(top_k);
        
        // compute softmax
        double sum_exp = 0;
        for (int j = 0; j < top_k; ++j)
        {
            logits_probs[j].first = std::exp(logits_probs[j].first);
            sum_exp += logits_probs[j].first;
        }
        for (int j = 0; j < top_k; ++j)
            logits_probs[j].first = logits_probs[j].first / sum_exp;

        std::vector<double> probs(logits_size, 0.0);
        for (int j = 0; j < top_k; j++)
        {
            const auto &prob_pair = logits_probs[j];
            probs[prob_pair.second] = prob_pair.first;
        }

        std::discrete_distribution dist(probs.begin(), probs.end());
        uint32_t maxi = dist(gen);
        if (maxi == eot_token) {
            // std::cerr << "<EOT>";
            break;
        }

        const int prev_token = (i == 0) ? 1 : tokens.back();
        std::cerr << tokenizer.decode(prev_token, maxi);

        tokens.push_back(maxi);
    }

    std::cerr << "\n";
}
