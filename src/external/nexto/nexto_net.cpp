// Derived from Necto (https://github.com/Rolv-Arild/Necto)
// by Rolv-Arild and the Necto contributors.
// Translated to C++ for buta-ppo. Included under the MIT License
// with permission; see THIRD_PARTY_NOTICES.md.

// buta-ppo/src/external/nexto/nexto_net.cpp
#include "nexto_net.hpp"
#include "nexto_bot.hpp"

#include <torch/csrc/jit/ir/ir.h>

#include <array>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace buta_ppo::external::nexto {

namespace {

constexpr int64_t DIM = 128;
constexpr int64_t HEADS = 4;
constexpr int64_t FF_DIM = 512;
constexpr int64_t BLOCKS = 2;
constexpr int64_t EMBED = 32;

// Where each of this module's top-level parts sits in the trace's module tree.
const std::array<std::pair<std::string, std::string>, 5> TRACE_PREFIXES = {{
    { "query_preprocess.", "net.earl.query_preprocess." },
    { "key_value_preprocess.", "net.earl.key_value_preprocess." },
    { "blocks.", "net.earl.blocks." },
    { "action_net.", "net.output.net." },
    { "emb_convertor.", "net.output.emb_convertor." },
}};

[[nodiscard]] std::string trace_name(const std::string& name) {
    for (const auto& [native, trace] : TRACE_PREFIXES) {
        if (name.rfind(native, 0) == 0) return trace + name.substr(native.size());
    }
    throw std::logic_error("NextoNet: no trace name for parameter '" + name + "'.");
}

/**
 * @brief The action table the trace feeds its output head: its only tensor constant.
 */
[[nodiscard]] torch::Tensor trace_action_table(const torch::jit::Module& model) {
    const torch::jit::Module output = model.attr("net").toModule().attr("output").toModule();
    torch::Tensor table;
    for (const torch::jit::Node* node : output.get_method("forward").graph()->nodes()) {
        if (node->kind() == c10::prim::Constant && node->output()->type()->isSubtypeOf(*c10::TensorType::get())) {
            if (table.defined()) {
                throw std::runtime_error("Nexto model: more than one constant in its output head.");
            }
            table = torch::jit::toIValue(node->output())->toTensor();
        }
    }
    if (!table.defined()) {
        throw std::runtime_error("Nexto model: no action table found in its output head.");
    }
    return table.to(torch::kFloat32);
}

} // namespace

AttentionImpl::AttentionImpl(int64_t dim, int64_t heads) : heads_(heads), head_dim_(dim / heads) {
    in_proj_weight_ = register_parameter("in_proj_weight", torch::empty({3 * dim, dim}));
    in_proj_bias_ = register_parameter("in_proj_bias", torch::empty({3 * dim}));
    out_proj_ = register_module("out_proj", torch::nn::Linear(dim, dim));
}

torch::Tensor AttentionImpl::forward(const torch::Tensor& query, const torch::Tensor& key_value) {
    const auto weights = in_proj_weight_.chunk(3);
    const auto biases = in_proj_bias_.chunk(3);
    const int64_t batch = query.size(0);

    // [B, L, D] -> [B, heads, L, head_dim]
    const auto split = [&](const torch::Tensor& x) {
        return x.view({batch, x.size(1), heads_, head_dim_}).transpose(1, 2);
    };
    const torch::Tensor q = split(torch::nn::functional::linear(query, weights[0], biases[0]))
                          / std::sqrt(static_cast<double>(head_dim_));
    const torch::Tensor k = split(torch::nn::functional::linear(key_value, weights[1], biases[1]));
    const torch::Tensor v = split(torch::nn::functional::linear(key_value, weights[2], biases[2]));

    const torch::Tensor attention = torch::softmax(torch::matmul(q, k.transpose(-2, -1)), -1);
    const torch::Tensor out = torch::matmul(attention, v).transpose(1, 2).reshape({batch, query.size(1), heads_ * head_dim_});
    return out_proj_->forward(out);
}

PerceiverBlockImpl::PerceiverBlockImpl(int64_t dim, int64_t heads, int64_t ff_dim) {
    attention_ = register_module("attention", Attention(dim, heads));
    linear1_ = register_module("linear1", torch::nn::Linear(dim, ff_dim));
    linear2_ = register_module("linear2", torch::nn::Linear(ff_dim, dim));
    norm1_ = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    norm2_ = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    norm3_ = register_module("norm3", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
}

torch::Tensor PerceiverBlockImpl::forward(const torch::Tensor& query, const torch::Tensor& entities) {
    const torch::Tensor x = query + attention_->forward(norm1_->forward(query), norm2_->forward(entities));
    return x + linear2_->forward(torch::relu(linear1_->forward(norm3_->forward(x))));
}

NextoNetImpl::NextoNetImpl() {
    using torch::nn::Linear;
    using torch::nn::ReLU;
    query_preprocess_ = register_module("query_preprocess",
        torch::nn::Sequential(Linear(static_cast<int64_t>(QUERY_SIZE), DIM), ReLU(), Linear(DIM, DIM), ReLU()));
    key_value_preprocess_ = register_module("key_value_preprocess",
        torch::nn::Sequential(Linear(static_cast<int64_t>(ENTITY_SIZE), DIM), ReLU(), Linear(DIM, DIM), ReLU()));
    blocks_ = register_module("blocks", torch::nn::ModuleList());
    for (int64_t b = 0; b < BLOCKS; b++) {
        blocks_->push_back(PerceiverBlock(DIM, HEADS, FF_DIM));
    }
    action_net_ = register_module("action_net",
        torch::nn::Sequential(Linear(static_cast<int64_t>(ACTION_SIZE), EMBED), ReLU(), Linear(EMBED, EMBED), ReLU(),
                              Linear(EMBED, EMBED), ReLU()));
    emb_convertor_ = register_module("emb_convertor", Linear(DIM, EMBED));
    action_embeddings_ = register_buffer("action_embeddings", torch::zeros({static_cast<int64_t>(NUM_ACTIONS), EMBED}));
}

void NextoNetImpl::load_from_torchscript(const torch::jit::Module& model) {
    std::unordered_map<std::string, torch::Tensor> trace_params;
    for (const auto& param : model.named_parameters(/*recurse=*/true)) {
        trace_params.emplace(param.name, param.value);
    }

    torch::NoGradGuard no_grad;
    std::set<std::string> used;
    for (auto& param : named_parameters(/*recurse=*/true)) {
        const std::string name = trace_name(param.key());
        const auto it = trace_params.find(name);
        if (it == trace_params.end()) {
            throw std::runtime_error("Nexto model: missing parameter '" + name + "'.");
        }
        if (it->second.sizes() != param.value().sizes()) {
            throw std::runtime_error("Nexto model: parameter '" + name + "' has shape " + c10::str(it->second.sizes())
                                     + ", expected " + c10::str(param.value().sizes()) + ".");
        }
        param.value().copy_(it->second);
        used.insert(name);
    }
    for (const auto& [name, value] : trace_params) {
        if (!used.contains(name)) {
            throw std::runtime_error("Nexto model: unexpected parameter '" + name + "'.");
        }
    }

    // The action embeddings the trace computes on every call, once.
    const torch::Tensor table = trace_action_table(model);
    torch::Tensor expected = torch::empty({static_cast<int64_t>(NUM_ACTIONS), static_cast<int64_t>(ACTION_SIZE)});
    for (size_t a = 0; a < NUM_ACTIONS; a++) {
        for (size_t i = 0; i < ACTION_SIZE; i++) {
            expected[static_cast<int64_t>(a)][static_cast<int64_t>(i)] = action_table()[a][i];
        }
    }
    if (!torch::equal(table, expected)) {
        throw std::runtime_error("Nexto model: its action table isn't agent.py's.");
    }
    action_embeddings_.copy_(action_net_->forward(table.to(action_embeddings_.device())));
}

torch::Tensor NextoNetImpl::forward(const torch::Tensor& query, const torch::Tensor& entities) {
    torch::Tensor q = query_preprocess_->forward(query);
    const torch::Tensor kv = key_value_preprocess_->forward(entities);
    for (const auto& block : *blocks_) {
        q = block->as<PerceiverBlock>()->forward(q, kv);
    }
    // [B, 1, 32] x [90, 32]^T -> [B, 1, 90], for the only query.
    const torch::Tensor embedding = emb_convertor_->forward(torch::relu(q));
    return torch::matmul(embedding, action_embeddings_.t()).select(1, 0);
}

} // namespace buta_ppo::external::nexto
