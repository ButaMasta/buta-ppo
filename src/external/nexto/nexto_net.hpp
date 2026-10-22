// Derived from Necto (https://github.com/Rolv-Arild/Necto)
// by Rolv-Arild and the Necto contributors.
// Translated to C++ for buta-ppo. Included under the MIT License
// with permission; see THIRD_PARTY_NOTICES.md.

// buta-ppo/src/external/nexto/nexto_net.hpp
#pragma once

#include <torch/torch.h>
#include <torch/script.h>

#include <cstdint>

namespace buta_ppo::external::nexto {

/**
 * @brief Multi-head attention of one query set over a key/value set, as `torch.nn.MultiheadAttention` in the trace.
 *
 * Parameter names match the trace's (`in_proj_weight`, `in_proj_bias`, `out_proj`), so its weights load by name.
 */
class AttentionImpl : public torch::nn::Module {
public:
    AttentionImpl(int64_t dim, int64_t heads);

    [[nodiscard]] torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& key_value);

private:
    int64_t heads_;
    int64_t head_dim_;
    torch::Tensor in_proj_weight_;
    torch::Tensor in_proj_bias_;
    torch::nn::Linear out_proj_{nullptr};
};
TORCH_MODULE(Attention);

/**
 * @brief The trace's `EARLPerceiverBlock`: attention of the query over the entities, then a feedforward, both residual.
 */
class PerceiverBlockImpl : public torch::nn::Module {
public:
    PerceiverBlockImpl(int64_t dim, int64_t heads, int64_t ff_dim);

    [[nodiscard]] torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& entities);

private:
    Attention attention_{nullptr};
    torch::nn::Linear linear1_{nullptr};
    torch::nn::Linear linear2_{nullptr};
    torch::nn::LayerNorm norm1_{nullptr};
    torch::nn::LayerNorm norm2_{nullptr};
    torch::nn::LayerNorm norm3_{nullptr};
};
TORCH_MODULE(PerceiverBlock);

/**
 * @brief Nexto's policy network, rebuilt natively from the botpack's TorchScript trace so it can run on the GPU.
 *
 * The trace pins its action table to the CPU. Here the action embeddings, which only ever see that fixed table, are
 * computed once at load, so the logits are one matmul against them. The trace's attention mask is added to the scores
 * and the bot always passes zeros, so it's left out.
 *
 * Takes the query `[B, 1, 32]` and entities `[B, N, 24]` and returns the logits of the 90 actions, `[B, 90]`.
 */
class NextoNetImpl : public torch::nn::Module {
public:
    NextoNetImpl();

    /**
     * @brief Copies every weight from the TorchScript model, checking names and shapes both ways, and computes the
     * action embeddings from its action table, which must match `nexto::action_table()`.
     */
    void load_from_torchscript(const torch::jit::Module& model);

    [[nodiscard]] torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& entities);

private:
    // The trace's parts, under the trace's names below `net.earl.` (preprocesses, blocks) and `net.output.`
    // (`net`, here `action_net`, and `emb_convertor`).
    torch::nn::Sequential query_preprocess_{nullptr};
    torch::nn::Sequential key_value_preprocess_{nullptr};
    torch::nn::ModuleList blocks_{nullptr};
    torch::nn::Sequential action_net_{nullptr};
    torch::nn::Linear emb_convertor_{nullptr};

    // The action embeddings, `[90, 32]`. A buffer, so it moves with the module.
    torch::Tensor action_embeddings_;
};
TORCH_MODULE(NextoNet);

} // namespace buta_ppo::external::nexto
