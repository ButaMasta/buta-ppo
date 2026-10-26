// Derived from Necto (https://github.com/Rolv-Arild/Necto)
// by Rolv-Arild and the Necto contributors.
// Translated to C++ for buta-ppo. Included under the MIT License
// with permission; see THIRD_PARTY_NOTICES.md.

// buta-ppo/src/external/nexto/nexto_bot.cpp
#include "nexto_bot.hpp"

#include <ATen/Context.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace buta_ppo::external {

namespace nexto {

namespace {

// Entity columns, from nexto_obs.py.
constexpr size_t IS_SELF = 0, IS_MATE = 1, IS_OPP = 2, IS_BALL = 3, IS_BOOST = 4;
constexpr size_t POS = 5, LIN_VEL = 8, FW = 11, UP = 14, ANG_VEL = 17;
constexpr size_t BOOST = 20, DEMO = 21, ON_GROUND = 22, HAS_FLIP = 23;
constexpr size_t ACTIONS = 24; // In the query only.

// nexto_obs.py's BOOST_LOCATIONS, in its order. Nexto was trained with these positions, pad order and heights (70 small,
// 73 big, which the BOOST column depends on). RLBot v5's field info has a different order and heights, which the v5
// Nexto patches back to these. The port uses them directly and never reads field info.
constexpr std::array<std::array<double, 3>, NUM_PADS> BOOST_LOCATIONS = {{
    {0.0, -4240.0, 70.0}, {-1792.0, -4184.0, 70.0}, {1792.0, -4184.0, 70.0}, {-3072.0, -4096.0, 73.0},
    {3072.0, -4096.0, 73.0}, {-940.0, -3308.0, 70.0}, {940.0, -3308.0, 70.0}, {0.0, -2816.0, 70.0},
    {-3584.0, -2484.0, 70.0}, {3584.0, -2484.0, 70.0}, {-1788.0, -2300.0, 70.0}, {1788.0, -2300.0, 70.0},
    {-2048.0, -1036.0, 70.0}, {0.0, -1024.0, 70.0}, {2048.0, -1036.0, 70.0}, {-3584.0, 0.0, 73.0},
    {-1024.0, 0.0, 70.0}, {1024.0, 0.0, 70.0}, {3584.0, 0.0, 73.0}, {-2048.0, 1036.0, 70.0},
    {0.0, 1024.0, 70.0}, {2048.0, 1036.0, 70.0}, {-1788.0, 2300.0, 70.0}, {1788.0, 2300.0, 70.0},
    {-3584.0, 2484.0, 70.0}, {3584.0, 2484.0, 70.0}, {0.0, 2816.0, 70.0}, {-940.0, 3310.0, 70.0},
    {940.0, 3308.0, 70.0}, {-3072.0, 4096.0, 73.0}, {3072.0, 4096.0, 73.0}, {-1792.0, 4184.0, 70.0},
    {1792.0, 4184.0, 70.0}, {0.0, 4240.0, 70.0},
}};

// RocketSim's soccar pad positions (x, y), which the bridge sorts by y, then x, for each size.
constexpr std::array<std::array<double, 2>, 6> ROCKETSIM_BIG_PADS = {{
    {-3584.0, 0.0}, {3584.0, 0.0}, {-3072.0, 4096.0}, {3072.0, 4096.0}, {-3072.0, -4096.0}, {3072.0, -4096.0},
}};
constexpr std::array<std::array<double, 2>, 28> ROCKETSIM_SMALL_PADS = {{
    {0.0, -4240.0}, {-1792.0, -4184.0}, {1792.0, -4184.0}, {-940.0, -3308.0}, {940.0, -3308.0}, {0.0, -2816.0},
    {-3584.0, -2484.0}, {3584.0, -2484.0}, {-1788.0, -2300.0}, {1788.0, -2300.0}, {-2048.0, -1036.0}, {0.0, -1024.0},
    {2048.0, -1036.0}, {-1024.0, 0.0}, {1024.0, 0.0}, {-2048.0, 1036.0}, {0.0, 1024.0}, {2048.0, 1036.0},
    {-1788.0, 2300.0}, {1788.0, 2300.0}, {-3584.0, 2484.0}, {3584.0, 2484.0}, {0.0, 2816.0}, {-940.0, 3308.0},
    {940.0, 3308.0}, {-1792.0, 4184.0}, {1792.0, 4184.0}, {0.0, 4240.0},
}};

/**
 * @brief Where each RLBot pad's cooldown is in the bridge's state: its index among the big or small pads.
 *
 * RLBot's order isn't the bridge's (e.g. (0, -1024) comes between the two pads at y = -1036), and RLBot's
 * positions differ by a few uu in places, so pads are matched by position.
 */
struct PadSource {
    bool big;
    size_t index;
};

template <size_t N>
[[nodiscard]] std::array<std::array<double, 2>, N> bridge_order(std::array<std::array<double, 2>, N> pads) {
    std::sort(pads.begin(), pads.end(), [](const auto& a, const auto& b) {
        return a[1] != b[1] ? a[1] < b[1] : a[0] < b[0];
    });
    return pads;
}

[[nodiscard]] std::array<PadSource, NUM_PADS> make_pad_sources() {
    const auto big = bridge_order(ROCKETSIM_BIG_PADS);
    const auto small = bridge_order(ROCKETSIM_SMALL_PADS);
    const auto find = [](const auto& pads, const std::array<double, 3>& loc) {
        for (size_t j = 0; j < pads.size(); j++) {
            if (std::abs(pads[j][0] - loc[0]) <= 5.0 && std::abs(pads[j][1] - loc[1]) <= 5.0) return j;
        }
        throw std::logic_error("Nexto: no RocketSim boost pad at an RLBot pad's position.");
    };

    std::array<PadSource, NUM_PADS> sources{};
    for (size_t i = 0; i < NUM_PADS; i++) {
        const bool is_big = BOOST_LOCATIONS[i][2] > 72.0;
        sources[i] = { is_big, is_big ? find(big, BOOST_LOCATIONS[i]) : find(small, BOOST_LOCATIONS[i]) };
    }
    return sources;
}

const std::array<PadSource, NUM_PADS> PAD_SOURCES = make_pad_sources();

[[nodiscard]] bool pad_active(const ffi::BoostPadState& pads, size_t rlbot_index) {
    const PadSource& source = PAD_SOURCES[rlbot_index];
    return (source.big ? pads.big[source.index] : pads.small[source.index]) == 0.0f;
}

/**
 * @brief bot.py's KICKOFF_CONTROLS, one action per tick.
 */
[[nodiscard]] std::vector<Action> make_kickoff_actions() {
    struct Segment {
        int ticks;
        Action action;
    };
    // throttle, steer, pitch, yaw, roll, jump, boost, handbrake
    const std::array<Segment, 7> segments = {{
        { 11 * 4, {1.0f,  0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f} },
        { 4 * 4,  {1.0f, -1.0f,  0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f} },
        { 2 * 4,  {1.0f,  0.0f,  0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f} },
        { 1 * 4,  {1.0f,  0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f} },
        { 1 * 4,  {1.0f,  0.0f, -0.7f, 0.8f, 0.0f, 1.0f, 1.0f, 0.0f} },
        { 13 * 4, {1.0f,  0.0f,  1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f} },
        { 10 * 4, {1.0f,  0.0f,  0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f} },
    }};

    std::vector<Action> actions;
    for (const Segment& segment : segments) {
        actions.insert(actions.end(), static_cast<size_t>(segment.ticks), segment.action);
    }
    return actions;
}

const std::vector<Action> KICKOFF_ACTIONS = make_kickoff_actions();

/**
 * @brief bot.py's `update_controls`.
 */
[[nodiscard]] ffi::CarControls to_controls(const Action& action) {
    ffi::CarControls controls{};
    controls.throttle = action[0];
    controls.steer = action[1];
    controls.pitch = action[2];
    controls.yaw = action[3];
    controls.roll = action[4];
    controls.jump = action[5] > 0.0f;
    controls.boost = action[6] > 0.0f;
    controls.handbrake = action[7] > 0.0f;
    return controls;
}

/**
 * @brief A car's forward and up, as the original gets them: rlgym_compat's matrix (columns forward, left, up) through
 * encode_gamestate's `rotation_to_quaternion` and nexto_obs's `_quats_to_rot_mtx`. The round trip renormalizes it.
 */
void round_trip_rotation(const float* forward, const float* up_in, std::array<double, 3>& fw, std::array<double, 3>& up) {
    const double f[3] = { forward[0], forward[1], forward[2] };
    const double u[3] = { up_in[0], up_in[1], up_in[2] };
    const double l[3] = { u[1] * f[2] - u[2] * f[1], u[2] * f[0] - u[0] * f[2], u[0] * f[1] - u[1] * f[0] }; // up x forward
    // m[row][col]
    const double m[3][3] = { { f[0], l[0], u[0] }, { f[1], l[1], u[1] }, { f[2], l[2], u[2] } };

    double q[4];
    const double trace = m[0][0] + m[1][1] + m[2][2];
    if (trace > 0) {
        double s = std::sqrt(trace + 1);
        q[0] = s * 0.5;
        s = 0.5 / s;
        q[1] = (m[2][1] - m[1][2]) * s;
        q[2] = (m[0][2] - m[2][0]) * s;
        q[3] = (m[1][0] - m[0][1]) * s;
    } else if (m[0][0] >= m[1][1] && m[0][0] >= m[2][2]) {
        const double s = std::sqrt(1 + m[0][0] - m[1][1] - m[2][2]);
        const double inv_s = 0.5 / s;
        q[1] = 0.5 * s;
        q[2] = (m[1][0] + m[0][1]) * inv_s;
        q[3] = (m[2][0] + m[0][2]) * inv_s;
        q[0] = (m[2][1] - m[1][2]) * inv_s;
    } else if (m[1][1] > m[2][2]) {
        const double s = std::sqrt(1 + m[1][1] - m[0][0] - m[2][2]);
        const double inv_s = 0.5 / s;
        q[1] = (m[0][1] + m[1][0]) * inv_s;
        q[2] = 0.5 * s;
        q[3] = (m[1][2] + m[2][1]) * inv_s;
        q[0] = (m[0][2] - m[2][0]) * inv_s;
    } else {
        const double s = std::sqrt(1 + m[2][2] - m[0][0] - m[1][1]);
        const double inv_s = 0.5 / s;
        q[1] = (m[0][2] + m[2][0]) * inv_s;
        q[2] = (m[1][2] + m[2][1]) * inv_s;
        q[3] = 0.5 * s;
        q[0] = (m[1][0] - m[0][1]) * inv_s;
    }
    // rotation_to_quaternion returns -q, and _quats_to_rot_mtx negates it back.
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double norm = w * w + x * x + y * y + z * z;
    if (norm == 0.0) {
        fw = {};
        up = {};
        return;
    }
    const double s = 1.0 / norm;
    fw = { 1.0 - 2.0 * s * (y * y + z * z), 2.0 * s * (x * y + z * w), 2.0 * s * (x * z - y * w) };
    up = { 2.0 * s * (x * z + y * w), 2.0 * s * (y * z - x * w), 1.0 - 2.0 * s * (x * x + y * y) };
}

/**
 * @brief agent.py's `act`: picks an action from the logits for a beta.
 */
[[nodiscard]] size_t pick_action(const float* logits, float beta, std::mt19937& rng) {
    if (beta == 1.0f) {
        return static_cast<size_t>(std::max_element(logits, logits + NUM_ACTIONS) - logits);
    }
    if (beta == -1.0f) {
        return static_cast<size_t>(std::min_element(logits, logits + NUM_ACTIONS) - logits);
    }

    // Sample from the logits scaled by log_3((1 + beta) / (1 - beta)), or uniformly at beta 0.
    const double scale = (beta == 0.0f) ? 0.0 : std::log((beta + 1.0) / (1.0 - beta)) / std::log(3.0);
    std::array<double, NUM_ACTIONS> weights{};
    double max_logit = -INFINITY;
    for (size_t a = 0; a < NUM_ACTIONS; a++) {
        weights[a] = static_cast<double>(logits[a]) * scale;
        max_logit = std::max(max_logit, weights[a]);
    }
    for (double& w : weights) {
        w = std::exp(w - max_logit);
    }
    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    return dist(rng);
}

} // namespace

const std::array<Action, NUM_ACTIONS>& action_table() {
    static const std::array<Action, NUM_ACTIONS> table = [] {
        std::array<Action, NUM_ACTIONS> actions{};
        size_t n = 0;
        // Ground
        for (const float throttle : {-1.0f, 0.0f, 1.0f}) {
            for (const float steer : {-1.0f, 0.0f, 1.0f}) {
                for (const float boost : {0.0f, 1.0f}) {
                    for (const float handbrake : {0.0f, 1.0f}) {
                        if (boost == 1.0f && throttle != 1.0f) continue;
                        // `throttle or boost`: boosting always throttles.
                        const float t = (throttle != 0.0f) ? throttle : boost;
                        actions[n++] = { t, steer, 0.0f, steer, 0.0f, 0.0f, boost, handbrake };
                    }
                }
            }
        }
        // Aerial
        for (const float pitch : {-1.0f, 0.0f, 1.0f}) {
            for (const float yaw : {-1.0f, 0.0f, 1.0f}) {
                for (const float roll : {-1.0f, 0.0f, 1.0f}) {
                    for (const float jump : {0.0f, 1.0f}) {
                        for (const float boost : {0.0f, 1.0f}) {
                            if (jump == 1.0f && yaw != 0.0f) continue;                        // Only need roll for sideflip
                            if (pitch == 0.0f && roll == 0.0f && jump == 0.0f) continue;  // Duplicate with ground
                            // Handbrake for potential wavedashes.
                            const bool handbrake = jump == 1.0f && (pitch != 0.0f || yaw != 0.0f || roll != 0.0f);
                            actions[n++] = { boost, yaw, pitch, yaw, roll, jump, boost, handbrake ? 1.0f : 0.0f };
                        }
                    }
                }
            }
        }
        if (n != NUM_ACTIONS) {
            throw std::logic_error("Nexto action table has " + std::to_string(n) + " actions.");
        }
        return actions;
    }();
    return table;
}

} // namespace nexto

using namespace nexto;

NextoBot::NextoBot(const std::string& model_path, const NextoOptions& options)
    : model_(std::make_shared<Model>()), options_(options) {
    if (!(options_.beta >= -1.0f && options_.beta <= 1.0f)) {
        throw std::invalid_argument("Nexto beta must be in [-1, 1].");
    }
    if (options_.torchscript && !options_.device.is_cpu()) {
        throw std::invalid_argument("Nexto's TorchScript file only runs on CPU.");
    }
    if (options_.device.is_cuda() && !torch::cuda::is_available()) {
        throw std::invalid_argument("Nexto: CUDA was asked for but isn't available.");
    }
    if (!std::filesystem::exists(model_path)) {
        throw std::invalid_argument("Nexto model not found at '" + model_path + "'.");
    }
    Model& m = *model_;
    m.device = options_.device;
    m.torchscript = options_.torchscript;
    try {
        m.module = torch::jit::load(model_path, torch::kCPU);
    } catch (const c10::Error& e) {
        throw std::runtime_error("Couldn't load the Nexto model from '" + model_path + "': " + e.what_without_backtrace());
    }
    m.module.eval();

    m.net->load_from_torchscript(m.module);
    m.net->eval();
    m.net->to(m.device);
    if (m.device.is_cuda()) {
        m.stream = c10::cuda::getStreamFromPool(false, m.device.index());
    }

    // Check the model's output on a 1v1 obs.
    std::vector<float> obs(obs_size(2), 0.0f);
    std::vector<float> logits(NUM_ACTIONS);
    infer(obs.data(), 1, 2, logits.data());
}

std::shared_ptr<NextoBot> NextoBot::with_options(const NextoOptions& options) const {
    if (!(options.beta >= -1.0f && options.beta <= 1.0f)) {
        throw std::invalid_argument("Nexto beta must be in [-1, 1].");
    }
    NextoOptions shared = options;
    shared.device = model_->device;
    shared.torchscript = model_->torchscript;
    return std::shared_ptr<NextoBot>(new NextoBot(model_, shared));
}

void NextoBot::infer(const float* obs, size_t count, size_t num_cars, float* decisions) {
    const std::lock_guard<std::mutex> lock(model_->mutex);
    const torch::InferenceMode inference_guard;

    const auto n = static_cast<int64_t>(count);
    const auto entities = static_cast<int64_t>(num_entities(num_cars));

    torch::Tensor logits;
    if (model_->device.is_cuda()) {
        logits = run_cuda(obs, n, entities);
    } else {
        const torch::Tensor all = torch::from_blob(const_cast<float*>(obs), {n, static_cast<int64_t>(obs_size(num_cars))});
        const torch::Tensor query = all.narrow(1, 0, QUERY_SIZE).reshape({n, 1, QUERY_SIZE});
        const torch::Tensor entity = all.narrow(1, QUERY_SIZE, entities * ENTITY_SIZE).reshape({n, entities, ENTITY_SIZE});
        logits = (model_->torchscript ? run_torchscript(query, entity) : model_->net->forward(query, entity)).contiguous();
    }

    if (logits.dim() != 2 || logits.size(0) != n || logits.size(1) != static_cast<int64_t>(NUM_ACTIONS)) {
        throw std::runtime_error("Nexto model gave logits of shape " + c10::str(logits.sizes()) + ", expected ["
                                 + std::to_string(n) + ", " + std::to_string(NUM_ACTIONS) + "].");
    }
    std::memcpy(decisions, logits.data_ptr<float>(), count * NUM_ACTIONS * sizeof(float));
}

torch::Tensor NextoBot::run_torchscript(const torch::Tensor& query, const torch::Tensor& entities) {
    // The mask is added to the attention scores. bot.py never pads, so it's all zeros.
    const torch::Tensor mask = torch::zeros({query.size(0), entities.size(1)});
    const auto out = model_->module.forward({c10::ivalue::Tuple::create({query, entities, mask})}).toTuple();
    return out->elements().at(0).toTensor();
}

torch::Tensor NextoBot::run_cuda(const float* obs, int64_t count, int64_t entities) {
    Model& m = *model_;
    const int64_t obs_floats = static_cast<int64_t>(QUERY_SIZE) + entities * static_cast<int64_t>(ENTITY_SIZE);
    const int64_t floats = count * obs_floats;
    const int64_t logit_floats = count * static_cast<int64_t>(NUM_ACTIONS);
    const c10::cuda::CUDAStreamGuard stream_guard(*m.stream);
    if (!m.host_obs.defined() || m.host_obs.numel() < floats) {
        m.host_obs = torch::empty({floats}, torch::TensorOptions().pinned_memory(true));
        m.device_obs = torch::empty({floats}, torch::TensorOptions().device(m.device));
    }
    if (!m.host_logits.defined() || m.host_logits.numel() < logit_floats) {
        m.host_logits = torch::empty({logit_floats}, torch::TensorOptions().pinned_memory(true));
    }

    std::memcpy(m.host_obs.data_ptr<float>(), obs, static_cast<size_t>(floats) * sizeof(float));
    const torch::Tensor all = m.device_obs.narrow(0, 0, floats).copy_(m.host_obs.narrow(0, 0, floats), true).view({count, obs_floats});
    const torch::Tensor query = all.narrow(1, 0, QUERY_SIZE).reshape({count, 1, QUERY_SIZE});
    const torch::Tensor entity = all.narrow(1, QUERY_SIZE, entities * ENTITY_SIZE).reshape({count, entities, ENTITY_SIZE});

    // TF32 changes Nexto's choices (on 0.4% of decisions from play), so its matmuls run in full fp32. 
    // cuBLAS's TF32 flag is process-wide and read when a matmul launches: this assumes no other thread 
    // launches eager fp32 matmuls meanwhile. Captured CUDA graphs keep the math they were captured with.
    struct Tf32Off {
        bool was = at::globalContext().allowTF32CuBLAS();
        Tf32Off() { at::globalContext().setAllowTF32CuBLAS(false); }
        ~Tf32Off() { at::globalContext().setAllowTF32CuBLAS(was); }
    } tf32_off;

    const torch::Tensor logits = m.host_logits.narrow(0, 0, logit_floats).view({count, static_cast<int64_t>(NUM_ACTIONS)});
    logits.copy_(m.net->forward(query, entity), true);
    m.stream->synchronize();
    return logits;
}

/**
 * @brief Nexto's network as CUDA graphs, for evaluation: see `make_batch_runner`.
 */
class NextoBot::GraphRunner final : public ExternalBatchRunner {
public:
    GraphRunner(std::shared_ptr<Model> model, const std::map<size_t, size_t>& max_batch) : model_(std::move(model)) {
        Model& m = *model_;
        const torch::InferenceMode inference_guard;
        // Captured without TF32, like `infer`. The flag is process-wide; this runs before evaluation plays.
        const bool tf32 = at::globalContext().allowTF32CuBLAS();
        at::globalContext().setAllowTF32CuBLAS(false);
        try {
            for (const auto& [num_cars, most] : max_batch) {
                if (most == 0) continue;
                Shape& s = shapes_.try_emplace(num_cars, c10::cuda::getStreamFromPool(false, m.device.index())).first->second;
                capture(s, num_cars, most);
            }
        } catch (...) {
            at::globalContext().setAllowTF32CuBLAS(tf32);
            throw;
        }
        at::globalContext().setAllowTF32CuBLAS(tf32);
    }

    [[nodiscard]] float* obs(size_t num_cars) override { return shapes_.at(num_cars).host_obs.data_ptr<float>(); }

    void launch(size_t num_cars, size_t count) override {
        Shape& s = shapes_.at(num_cars);
        // Replay launches on the current stream.
        const c10::cuda::CUDAStreamGuard stream_guard(s.stream);
        if (count <= s.batches.back()) {
            size_t k = 0;
            while (s.batches[k] < count) k++;
            s.graphs[k]->replay();
            return;
        }
        // Larger batches run as chunks of the largest graph, one after another.
        const size_t chunks = (count + s.batches.back() - 1) / s.batches.back();
        s.graphs.back()->replay();
        for (size_t c = 1; c < chunks; c++) s.chunk_graphs[c - 1]->replay();
    }

    [[nodiscard]] const float* wait(size_t num_cars) override {
        Shape& s = shapes_.at(num_cars);
        s.stream.synchronize();
        return s.host_logits.data_ptr<float>();
    }

private:
    // The largest graph's batch. Larger batches run in chunks, which keeps the graphs' GPU memory bounded.
    static constexpr size_t MAX_GRAPH_BATCH = 1024;

    struct Shape {
        explicit Shape(c10::cuda::CUDAStream stream) : stream(stream) {}
        c10::cuda::CUDAStream stream;
        at::cuda::MempoolId_t pool{}; // Its own: car counts replay at the same time. Shared by its graphs, which don't.
        torch::Tensor host_obs;    // Pinned.
        torch::Tensor device_obs;
        torch::Tensor host_logits; // Pinned.
        std::vector<size_t> batches; // Each graph's batch size, ascending, all from the start of the buffers.
        std::vector<std::unique_ptr<at::cuda::CUDAGraph>> graphs;
        // The largest batch again, for each later chunk of the buffers.
        std::vector<std::unique_ptr<at::cuda::CUDAGraph>> chunk_graphs;
    };
    std::shared_ptr<Model> model_;
    std::map<size_t, Shape> shapes_;

    void capture(Shape& s, size_t num_cars, size_t most) {
        Model& m = *model_;
        const auto entities = static_cast<int64_t>(num_entities(num_cars));
        const int64_t obs_floats = static_cast<int64_t>(QUERY_SIZE) + entities * static_cast<int64_t>(ENTITY_SIZE);
        size_t top = 1;
        while (top < most && top < MAX_GRAPH_BATCH) top *= 2;
        const size_t chunks = (most + top - 1) / top;
        const auto rows = static_cast<int64_t>(chunks * top);
        const auto pinned = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
        s.host_obs = torch::zeros({rows * obs_floats}, pinned);
        s.device_obs = torch::zeros({rows * obs_floats}, torch::TensorOptions().device(m.device));
        s.host_logits = torch::zeros({rows * static_cast<int64_t>(NUM_ACTIONS)}, pinned);
        s.pool = at::cuda::graph_pool_handle();

        // Capture runs on the shape's stream, after any work already queued on the current one.
        {
            at::cuda::CUDAEvent ready;
            ready.record(at::cuda::getCurrentCUDAStream());
            ready.block(s.stream);
        }
        const c10::cuda::CUDAStreamGuard stream_guard(s.stream);
        // Uploads `n` obs from row `first`, runs the network on them and downloads their logits to the same rows.
        const auto capture_one = [&](int64_t first, int64_t n) {
            const auto body = [&] {
                const torch::Tensor all = s.device_obs.narrow(0, first * obs_floats, n * obs_floats)
                                              .copy_(s.host_obs.narrow(0, first * obs_floats, n * obs_floats), true)
                                              .view({n, obs_floats});
                const torch::Tensor query = all.narrow(1, 0, QUERY_SIZE).reshape({n, 1, QUERY_SIZE});
                const torch::Tensor entity = all.narrow(1, QUERY_SIZE, entities * ENTITY_SIZE).reshape({n, entities, ENTITY_SIZE});
                s.host_logits.narrow(0, first * static_cast<int64_t>(NUM_ACTIONS), n * static_cast<int64_t>(NUM_ACTIONS))
                    .view({n, static_cast<int64_t>(NUM_ACTIONS)})
                    .copy_(m.net->forward(query, entity), true);
            };
            // Warmup runs lazy initialization (e.g. cuBLAS workspaces) outside of capture.
            for (int i = 0; i < 3; i++) body();
            s.stream.synchronize();

            auto graph = std::make_unique<at::cuda::CUDAGraph>();
            graph->capture_begin(s.pool);
            body();
            graph->capture_end();
            return graph;
        };
        for (size_t batch = 1; batch <= top; batch *= 2) {
            s.batches.push_back(batch);
            s.graphs.push_back(capture_one(0, static_cast<int64_t>(batch)));
        }
        for (size_t c = 1; c < chunks; c++) {
            s.chunk_graphs.push_back(capture_one(static_cast<int64_t>(c * top), static_cast<int64_t>(top)));
        }
    }
};

std::unique_ptr<ExternalBatchRunner> NextoBot::make_batch_runner(const std::map<size_t, size_t>& max_batch) {
    if (!model_->device.is_cuda()) {
        return ExternalBot::make_batch_runner(max_batch);
    }
    return std::make_unique<GraphRunner>(model_, max_batch);
}

std::unique_ptr<ExternalController> NextoBot::make_controller(uint32_t car_id) const {
    return std::make_unique<NextoController>(car_id, options_);
}

NextoController::NextoController(uint32_t car_id, const NextoOptions& options) : car_id_(car_id), options_(options) {}

void NextoController::observe(const ffi::ArenaState& state, const GameInfo& info) {
    info_ = info;
    for (uint32_t c = 0; c < state.num_cars; c++) {
        ticks_off_ground_[c] = state.cars[c].is_on_ground ? 0 : ticks_off_ground_[c] + 1;
    }
}

bool NextoController::on_ground(uint32_t car_id, const ffi::ArenaState& state) const {
    return state.cars[car_id].is_on_ground || ticks_off_ground_[car_id] <= 6;
}

void NextoController::build_obs(const ffi::ArenaState& state, float* out) const {
    const size_t num_cars = state.num_cars;
    const size_t entities = num_entities(num_cars);
    const int self_team = state.cars[car_id_].team;

    // bot.py's player order: self, then teammates, then opponents, each in car index order.
    std::array<uint32_t, MAX_CARS> order{};
    size_t players = 0;
    order[players++] = car_id_;
    for (uint32_t c = 0; c < num_cars; c++) {
        if (c != car_id_ && state.cars[c].team == self_team) order[players++] = c;
    }
    for (uint32_t c = 0; c < num_cars; c++) {
        if (state.cars[c].team != self_team) order[players++] = c;
    }

    // Built in double like nexto_obs.py's numpy arrays, and only cast to float at the end.
    std::array<std::array<double, ENTITY_SIZE>, num_entities(MAX_CARS)> kv{};
    const auto put3 = [](std::array<double, ENTITY_SIZE>& row, size_t col, const float* v) {
        row[col] = v[0];
        row[col + 1] = v[1];
        row[col + 2] = v[2];
    };

    // Players.
    for (size_t i = 0; i < num_cars; i++) {
        const uint32_t c = order[i];
        const ffi::CarState& car = state.cars[c];
        auto& row = kv[i];
        row[IS_SELF] = (i == 0) ? 1.0 : 0.0;
        row[IS_MATE] = 1.0 - car.team; // Default team is blue.
        row[IS_OPP] = car.team;
        put3(row, POS, car.phys.pos);
        put3(row, LIN_VEL, car.phys.vel);
        std::array<double, 3> fw{}, up{};
        round_trip_rotation(car.phys.rot_mat[0], car.phys.rot_mat[2], fw, up);
        std::copy(fw.begin(), fw.end(), row.begin() + FW);
        std::copy(up.begin(), up.end(), row.begin() + UP);
        put3(row, ANG_VEL, car.phys.ang_vel);
        row[BOOST] = static_cast<double>(car.boost) / 100.0;
        row[DEMO] = car.is_demoed ? 1.0 : 0.0;
        row[ON_GROUND] = on_ground(c, state) ? 1.0 : 0.0;
        row[HAS_FLIP] = (car.has_double_jumped || car.has_flipped) ? 0.0 : 1.0;
    }

    // Ball.
    {
        auto& row = kv[num_cars];
        row[IS_BALL] = 1.0;
        put3(row, POS, state.ball.phys.pos);
        put3(row, LIN_VEL, state.ball.phys.vel);
        put3(row, ANG_VEL, state.ball.phys.ang_vel);
    }

    // Boost pads. Their active flag goes in the DEMO column.
    for (size_t p = 0; p < NUM_PADS; p++) {
        auto& row = kv[num_cars + 1 + p];
        row[IS_BOOST] = 1.0;
        row[POS] = BOOST_LOCATIONS[p][0];
        row[POS + 1] = BOOST_LOCATIONS[p][1];
        row[POS + 2] = BOOST_LOCATIONS[p][2];
        row[BOOST] = 0.12 + 0.88 * (BOOST_LOCATIONS[p][2] > 72.0 ? 1.0 : 0.0);
        row[DEMO] = pad_active(state.boost_pads, p) ? 1.0 : 0.0;
    }

    for (size_t e = 0; e < entities; e++) {
        auto& row = kv[e];
        // Orange sees the field rotated 180 degrees, with teams swapped.
        if (self_team == static_cast<int>(ffi::Team::Orange)) {
            for (size_t col = POS; col < BOOST; col += 3) {
                row[col] = -row[col];
                row[col + 1] = -row[col + 1];
            }
            std::swap(row[IS_MATE], row[IS_OPP]);
        }
        for (size_t col = POS; col < FW; col++) {
            row[col] /= 2300.0;
        }
        for (size_t col = ANG_VEL; col < BOOST; col++) {
            row[col] /= 5.5;
        }
    }

    // The query is the own car's absolute entity row, plus the previous action.
    std::array<double, QUERY_SIZE> q{};
    std::copy(kv[0].begin(), kv[0].end(), q.begin());
    std::copy(action_.begin(), action_.end(), q.begin() + ACTIONS);

    // Every entity relative to the own car: positions offset, then all vectors rotated by its heading.
    const double theta = std::atan2(q[FW], q[FW + 1]);
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    for (size_t e = 0; e < entities; e++) {
        auto& row = kv[e];
        for (size_t col = POS; col < POS + 3; col++) {
            row[col] -= q[col];
        }
        for (size_t col = POS; col < BOOST; col += 3) {
            const double x = row[col];
            const double y = row[col + 1];
            row[col] = ct * x - st * y;
            row[col + 1] = st * x + ct * y;
        }
    }

    for (size_t i = 0; i < QUERY_SIZE; i++) {
        out[i] = static_cast<float>(q[i]);
    }
    float* entity_out = out + QUERY_SIZE;
    for (size_t e = 0; e < entities; e++) {
        for (size_t col = 0; col < ENTITY_SIZE; col++) {
            entity_out[e * ENTITY_SIZE + col] = static_cast<float>(kv[e][col]);
        }
    }
}

void NextoController::request_decision() {
    pending_beta_ = (options_.stochastic_kickoffs && info_.kickoff_pause) ? 0.5f : options_.beta;
    pending_ = true;
    pending_overwritten_ = false;
    update_action_ = false;
}

void NextoController::set_decision(const float* decision, std::mt19937& rng) {
    if (pending_) {
        // Set late: as if set at its packet, before everything since.
        last_beta_ = pending_beta_;
        last_action_index_ = pick_action(decision, pending_beta_, rng);
        if (!pending_overwritten_) {
            action_ = action_table()[last_action_index_];
        }
        pending_ = false;
        return;
    }
    float beta = options_.beta;
    if (options_.stochastic_kickoffs && info_.kickoff_pause) {
        beta = 0.5f;
    }
    last_beta_ = beta;
    last_action_index_ = pick_action(decision, beta, rng);
    action_ = action_table()[last_action_index_];
    update_action_ = false;
}

ffi::CarControls NextoController::tick(const ffi::ArenaState& state) {
    ticks_ += 1;
    if (ticks_ >= TICK_SKIP - 1) {
        controls_ = to_controls(action_);
    }
    if (ticks_ >= TICK_SKIP) {
        ticks_ = 0;
        update_action_ = true;
    }
    if (options_.hardcoded_kickoffs) {
        maybe_do_kickoff(state);
    }
    return controls_;
}

void NextoController::maybe_do_kickoff(const ffi::ArenaState& state) {
    if (!info_.kickoff_pause) {
        kickoff_index_ = -1;
        return;
    }

    if (kickoff_index_ >= 0) {
        kickoff_index_ += 1;
    } else if (kickoff_index_ == -1) {
        kickoff_index_ = is_kickoff_taker(state) ? 0 : -2;
    }

    if (kickoff_index_ >= 0 && kickoff_index_ < static_cast<int>(KICKOFF_ACTIONS.size())
        && state.ball.phys.pos[1] == 0.0f) {
        action_ = KICKOFF_ACTIONS[static_cast<size_t>(kickoff_index_)];
        controls_ = to_controls(action_);
        if (pending_) pending_overwritten_ = true;
    }
}

bool NextoController::is_kickoff_taker(const ffi::ArenaState& state) const {
    const double ball_x = state.ball.phys.pos[0];
    const double ball_y = state.ball.phys.pos[1];
    std::array<double, MAX_CARS> distances{};
    double min_distance = INFINITY;
    for (uint32_t c = 0; c < state.num_cars; c++) {
        distances[c] = std::hypot(state.cars[c].phys.pos[0] - ball_x, state.cars[c].phys.pos[1] - ball_y);
        min_distance = std::min(min_distance, distances[c]);
    }

    const double own = distances[car_id_];
    if (std::abs(min_distance - own) > 10.0) {
        return false;
    }

    // Among teammates equally close, the one on the left goes.
    const int team = state.cars[car_id_].team;
    const double own_x = state.cars[car_id_].phys.pos[0];
    for (uint32_t c = 0; c < state.num_cars; c++) {
        if (c == car_id_ || state.cars[c].team != team || std::abs(distances[c] - own) > 10.0) continue;
        const double x = state.cars[c].phys.pos[0];
        const bool is_left = (team == static_cast<int>(ffi::Team::Blue)) ? x < own_x : x > own_x;
        if (!is_left) {
            return false;
        }
    }
    return true;
}

} // namespace buta_ppo::external
