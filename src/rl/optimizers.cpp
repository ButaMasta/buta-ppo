// buta-ppo/src/rl/optimizers.cpp
#include "optimizers.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace buta_ppo::rl {

namespace {

std::string join_names(const std::vector<std::string>& names) {
    std::string out;
    for (const auto& name : names) {
        out += (out.empty() ? "" : ", ") + name;
    }
    return out;
}

/**
 * @brief Adam / AdamW via the fused multi-tensor kernels.
 *
 * Step counts live on the device so bias correction advances on every graph replay. So does the learning rate, through
 * the kernels' tensor-lr variant.
 */
class FusedAdamOptimizer final : public GraphSafeOptimizer {
private:
    OptimizerSpec spec_;
    std::vector<std::string> names_;
    std::vector<torch::Tensor> params_;
    torch::Tensor lr_scale_;
    torch::Tensor lr_; // spec_.lr * lr_scale_, refilled every step.
    std::vector<torch::Tensor> grads_; // Refilled each step within reserved capacity.
    std::vector<torch::Tensor> exp_avgs_;
    std::vector<torch::Tensor> exp_avg_sqs_;
    std::vector<torch::Tensor> state_steps_;

public:
    FusedAdamOptimizer(const OptimizerSpec& spec, std::vector<std::string> names, std::vector<torch::Tensor> params,
                       torch::Tensor lr_scale)
        : spec_(spec), names_(std::move(names)), params_(std::move(params)), lr_scale_(std::move(lr_scale)) {
        lr_ = torch::empty({}, lr_scale_.options());
        for (const auto& p : params_) {
            exp_avgs_.push_back(torch::zeros_like(p));
            exp_avg_sqs_.push_back(torch::zeros_like(p));
            state_steps_.push_back(torch::zeros({}, p.options().dtype(torch::kFloat32)));
        }
        grads_.reserve(params_.size());
    }

    void step() override {
        torch::NoGradGuard no_grad;
        grads_.clear();
        for (const auto& p : params_) {
            grads_.push_back(p.grad());
        }

        torch::_foreach_add_(state_steps_, 1);
        // In float32, as the kernels use a scalar lr: at scale 1 it's exactly spec_.lr.
        lr_.copy_(lr_scale_).mul_(spec_.lr);
        if (spec_.type == OptimizerType::AdamW) {
            at::_fused_adamw_(params_, grads_, exp_avgs_, exp_avg_sqs_, {}, state_steps_, lr_, spec_.beta1,
                              spec_.beta2, spec_.weight_decay, spec_.eps, /*amsgrad=*/false, /*maximize=*/false);
        } else {
            at::_fused_adam_(params_, grads_, exp_avgs_, exp_avg_sqs_, {}, state_steps_, lr_, spec_.beta1,
                             spec_.beta2, spec_.weight_decay, spec_.eps, /*amsgrad=*/false, /*maximize=*/false);
        }
    }

    std::vector<std::pair<std::string, torch::Tensor>> named_state() const override {
        std::vector<std::pair<std::string, torch::Tensor>> state;
        for (size_t i = 0; i < params_.size(); ++i) {
            state.emplace_back("exp_avg." + names_[i], exp_avgs_[i]);
            state.emplace_back("exp_avg_sq." + names_[i], exp_avg_sqs_[i]);
            state.emplace_back("step." + names_[i], state_steps_[i]);
        }
        return state;
    }

    std::string describe() const override {
        std::ostringstream out;
        out << (spec_.type == OptimizerType::AdamW ? "AdamW" : "Adam") << "(lr=" << spec_.lr
            << ", wd=" << spec_.weight_decay << ", betas=(" << spec_.beta1 << ", " << spec_.beta2
            << "), eps=" << spec_.eps << ") on " << names_.size() << " tensors: " << join_names(names_);
        return out.str();
    }
};

/**
 * @brief Muon: momentum orthogonalized by Newton-Schulz iteration, for 2D hidden weight matrices.
 *
 * Follows torch.optim.Muon. Every op runs on the device with fixed shapes, and there is no step counter.
 */
class MuonOptimizer final : public GraphSafeOptimizer {
private:
    OptimizerSpec spec_;
    std::vector<std::string> names_;
    std::vector<torch::Tensor> params_;
    std::vector<torch::Tensor> momentum_buffers_;
    std::vector<double> adjusted_lrs_; // lr scaled for each matrix's shape.
    torch::Tensor lr_scale_;
    // Refilled every step from lr_scale_: each matrix's negated step size and the weight decay factor. In float64, so
    // they round to float32 in the kernels as the scalars they replace did, and scale 1 updates exactly as before.
    std::vector<torch::Tensor> neg_lrs_;
    torch::Tensor decay_;

    // Approximates the nearest semi-orthogonal matrix to `g` with a quintic Newton-Schulz iteration in bf16.
    static torch::Tensor newton_schulz(const torch::Tensor& g, int steps) {
        constexpr double a = 3.4445, b = -4.7750, c = 2.0315;

        torch::Tensor x = g.to(torch::kBFloat16);
        const bool transposed = g.size(0) > g.size(1);
        if (transposed) x = x.mT();
        x = x / x.norm().clamp_min(1e-7);

        for (int i = 0; i < steps; ++i) {
            const torch::Tensor gram = torch::mm(x, x.mT());
            const torch::Tensor gram_update = torch::addmm(gram, gram, gram, /*beta=*/b, /*alpha=*/c);
            x = torch::addmm(x, gram_update, x, /*beta=*/a);
        }
        return transposed ? x.mT() : x;
    }

public:
    MuonOptimizer(const OptimizerSpec& spec, std::vector<std::string> names, std::vector<torch::Tensor> params,
                  torch::Tensor lr_scale)
        : spec_(spec), names_(std::move(names)), params_(std::move(params)), lr_scale_(std::move(lr_scale)) {
        const auto scalar_opts = lr_scale_.options().dtype(torch::kFloat64);
        decay_ = torch::empty({}, scalar_opts);
        for (size_t i = 0; i < params_.size(); ++i) {
            const torch::Tensor& p = params_[i];
            TORCH_CHECK(p.dim() == 2, "Muon requires 2D weight matrices, but '", names_[i], "' has ", p.dim(), " dims.");
            momentum_buffers_.push_back(torch::zeros_like(p));

            const double rows = static_cast<double>(p.size(0));
            const double cols = static_cast<double>(p.size(1));
            const double ratio = spec_.muon_lr_scale == MuonLrScale::MatchAdamW
                ? 0.2 * std::sqrt(std::max(rows, cols))
                : std::sqrt(std::max(1.0, rows / cols));
            adjusted_lrs_.push_back(spec_.lr * ratio);
            neg_lrs_.push_back(torch::empty({}, scalar_opts));
        }
    }

    void step() override {
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < params_.size(); ++i) {
            torch::Tensor& p = params_[i];
            const torch::Tensor& g = p.grad();
            torch::Tensor& buf = momentum_buffers_[i];

            buf.lerp_(g, 1.0 - spec_.momentum);
            const torch::Tensor update = spec_.nesterov ? g.lerp(buf, spec_.momentum) : buf;
            const torch::Tensor ortho = newton_schulz(update, spec_.ns_steps);

            if (spec_.weight_decay != 0.0f) {
                // 1 - lr * scale * wd.
                decay_.copy_(lr_scale_).mul_(-static_cast<double>(spec_.lr) * spec_.weight_decay).add_(1.0);
                p.mul_(decay_);
            }
            // p -= adjusted lr * scale * ortho. addcmul computes in float32 like add's alpha, where multiplying ortho by
            // the lr first would round the step to ortho's bf16.
            neg_lrs_[i].copy_(lr_scale_).mul_(-adjusted_lrs_[i]);
            p.addcmul_(ortho, neg_lrs_[i]);
        }
    }

    std::vector<std::pair<std::string, torch::Tensor>> named_state() const override {
        std::vector<std::pair<std::string, torch::Tensor>> state;
        for (size_t i = 0; i < params_.size(); ++i) {
            state.emplace_back("momentum_buffer." + names_[i], momentum_buffers_[i]);
        }
        return state;
    }

    std::string describe() const override {
        std::ostringstream out;
        out << "Muon(lr=" << spec_.lr << ", wd=" << spec_.weight_decay << ", momentum=" << spec_.momentum
            << ", nesterov=" << (spec_.nesterov ? "true" : "false") << ", ns_steps=" << spec_.ns_steps
            << ", lr_scale=" << (spec_.muon_lr_scale == MuonLrScale::MatchAdamW ? "match_adamw" : "original")
            << ") on " << names_.size() << " tensors: " << join_names(names_);
        return out.str();
    }
};

const OptimizerSpec& spec_for(const ParamInfo& info, const OptimizerConfig& config) {
    const SectionOptimizer& section = info.section == ParamSection::Shared ? config.shared
                                    : info.section == ParamSection::Actor  ? config.actor
                                                                           : config.critic;
    switch (info.kind) {
        case ParamKind::HiddenWeight: return section.hidden_weights;
        case ParamKind::IOWeight:     return section.io_weights;
        case ParamKind::Vector:       return section.vectors;
    }
    throw std::logic_error("Unhandled ParamKind.");
}

void validate(const OptimizerSpec& spec, const ParamInfo& info) {
    if (!(spec.lr > 0.0f)) {
        throw std::invalid_argument("Optimizer lr must be positive (parameter '" + info.name + "').");
    }
    if (spec.type == OptimizerType::Muon) {
        if (info.kind != ParamKind::HiddenWeight) {
            throw std::invalid_argument("Muon only supports hidden-to-hidden weight matrices; '" + info.name +
                                        "' must use Adam or AdamW (set its io_weights/vectors spec).");
        }
        if (spec.ns_steps < 1) {
            throw std::invalid_argument("Muon ns_steps must be at least 1 (parameter '" + info.name + "').");
        }
    }
}

} // namespace

float LrSchedule::scale_at(int64_t steps) const {
    if (shape == Shape::Constant) return start_scale;
    if (steps <= start_steps) return start_scale;
    if (steps >= end_steps) return end_scale;

    const double t = static_cast<double>(steps - start_steps) / static_cast<double>(end_steps - start_steps);
    const double from = start_scale;
    const double to = end_scale;
    switch (shape) {
        case Shape::Linear:      return static_cast<float>(from + (to - from) * t);
        case Shape::Exponential: return static_cast<float>(from * std::pow(to / from, t));
        case Shape::Cosine:      return static_cast<float>(to + (from - to) * 0.5 * (1.0 + std::cos(M_PI * t)));
        case Shape::Constant:    break;
    }
    return start_scale;
}

void LrSchedule::validate() const {
    const auto positive = [](float v) { return std::isfinite(v) && v > 0.0f; };
    if (!positive(start_scale) || (shape != Shape::Constant && !positive(end_scale))) {
        throw std::invalid_argument("LrSchedule scales must be positive and finite.");
    }
    if (shape != Shape::Constant && (start_steps < 0 || end_steps <= start_steps)) {
        throw std::invalid_argument("An LrSchedule that changes must start at 0 steps or later and end after it starts (" +
                                    std::to_string(start_steps) + " to " + std::to_string(end_steps) + ").");
    }
}

std::vector<std::unique_ptr<GraphSafeOptimizer>> build_optimizers(
    const std::vector<ParamInfo>& layout,
    const OptimizerConfig& config,
    const torch::Tensor& lr_scale
) {
    TORCH_CHECK(lr_scale.dim() == 0 && lr_scale.scalar_type() == torch::kFloat32,
                "lr_scale must be a 0-dim float32 tensor.");
    // Bucket parameters by identical spec, in order of first appearance, so each spec is one optimizer.
    struct Bucket {
        OptimizerSpec spec;
        std::vector<std::string> names;
        std::vector<torch::Tensor> params;
    };
    std::vector<Bucket> buckets;

    for (const auto& info : layout) {
        const OptimizerSpec& spec = spec_for(info, config);
        validate(spec, info);

        auto it = std::find_if(buckets.begin(), buckets.end(), [&](const Bucket& b) { return b.spec == spec; });
        if (it == buckets.end()) {
            buckets.push_back({spec, {}, {}});
            it = std::prev(buckets.end());
        }
        it->names.push_back(info.name);
        it->params.push_back(info.tensor);
    }

    std::vector<std::unique_ptr<GraphSafeOptimizer>> optimizers;
    for (auto& b : buckets) {
        if (b.spec.type == OptimizerType::Muon) {
            optimizers.push_back(std::make_unique<MuonOptimizer>(b.spec, std::move(b.names), std::move(b.params), lr_scale));
        } else {
            optimizers.push_back(std::make_unique<FusedAdamOptimizer>(b.spec, std::move(b.names), std::move(b.params),
                                                                      lr_scale));
        }
    }
    return optimizers;
}

} // namespace buta_ppo::rl
