// buta-ppo/src/rl/ppo_runner.cpp
#include "ppo_runner.hpp"
#include "rollout_buffer.hpp"
#include "opponent_pool.hpp"
#include "tensorboard_logger.h"
#include "env/vec_env.hpp"
#include "util/thread_affinity.hpp"
#include "util/cpu_topology.hpp"
#include "util/file_utils.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <limits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace buta_ppo::rl {

namespace {
// File names inside each `<bot name>_<steps>/` checkpoint directory.
constexpr const char* MODEL_FILE = "model.pt";
constexpr const char* OPTIMIZER_FILE = "optimizer.pt";
// Suffix of the PFSP pool's directory, `<bot name>_pool/`, next to the checkpoints.
// Checkpoint loading skips it, since "pool" isn't a step count.
constexpr const char* POOL_DIR_SUFFIX = "_pool";
// Suffix of the Evaluation pool's directory.
constexpr const char* EVAL_DIR_SUFFIX = "_eval";

/**
 * @brief Converts an episode limit in seconds to physics ticks, 120 per second. 0 stays 0, which turns it off.
 *
 * @throws std::invalid_argument If the limit is negative, not finite, too large, or shorter than one tick.
 */
int limit_ticks(float seconds, const std::string& name) {
    constexpr double TICKS_PER_SECOND = 120.0;
    if (!std::isfinite(seconds) || seconds < 0.0f) {
        throw std::invalid_argument(name + " must be finite and at least 0 (0 turns it off).");
    }
    const double ticks = std::round(static_cast<double>(seconds) * TICKS_PER_SECOND);
    if (ticks > static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(name + " is too large. Set it to 0 to turn the limit off.");
    }
    if (seconds > 0.0f && ticks == 0.0) {
        throw std::invalid_argument(name + " must be 0 (off) or at least one physics tick, 1/120 of a second.");
    }
    return static_cast<int>(ticks);
}
} // namespace

size_t learner_agents_for(size_t envs, const RunnerConfig& config) {
    const std::vector<size_t> counts = compute_env_counts(envs, config.match_distributions);
    size_t learners = 0;
    for (size_t d = 0; d < counts.size(); d++) {
        const MatchDistribution& dist = config.match_distributions[d];
        // As VecEnv assigns them: a share of the even-team envs, where the past version plays one team.
        size_t opponent_envs = 0;
        if (config.pfsp_cfg.enabled() && !config.render && dist.blue_players == dist.orange_players) {
            opponent_envs = static_cast<size_t>(std::lround(config.pfsp_cfg.envs * static_cast<float>(counts[d])));
        }
        learners += counts[d] * dist.total_players() - opponent_envs * dist.blue_players;
    }
    return learners;
}

size_t RunnerConfig::total_envs() const {
    if (num_envs > 0) return num_envs;
    if (target_steps_per_agent == 0) {
        throw std::invalid_argument("Set num_envs or target_steps_per_agent, since both are 0.");
    }

    // Steps per agent only shrink as envs grow, so stop once they fall below a tenth of the target.
    size_t best = 2;
    double best_error = std::numeric_limits<double>::infinity();
    for (size_t envs = 2; envs < 1'000'000; envs++) {
        const size_t learners = learner_agents_for(envs, *this);
        if (learners == 0) continue;
        const double steps_per_agent = static_cast<double>(target_steps_per_update) / static_cast<double>(learners);
        const double error = std::abs(std::log(steps_per_agent / static_cast<double>(target_steps_per_agent)));
        if (error < best_error) {
            best_error = error;
            best = envs;
        }
        if (steps_per_agent < 0.1 * static_cast<double>(target_steps_per_agent)) break;
    }
    return best;
}

PPORunner::PPORunner(const RunnerConfig& config) : PPORunner(config, false) {}

PPORunner::PPORunner(const RunnerConfig& config, bool benchmark_child)
    : config_(config), device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU), benchmark_child_(benchmark_child) {
    
    if (torch::cuda::is_available()) {
        if (!benchmark_child_) {
            std::cout << "CUDA detected. Running on GPU.\n";
        }
        at::globalContext().setUserEnabledCuDNN(true);
        at::globalContext().setBenchmarkCuDNN(true);

        // TF32 matmuls for the fp32 rollout inference (training matmuls run in bf16 under autocast).
        // TF32 keeps fp32 range and accumulation with a 10-bit mantissa, still above bf16's 7.
        at::globalContext().setAllowTF32CuBLAS(true);
    } else {
        throw std::runtime_error("CUDA not found. CPU is currently not supported.");
    }

    if (config_.benchmark && config_.render) {
        throw std::invalid_argument("benchmark and render can't both be set.");
    }
    if (config_.calibrate_nexto_ladder && (config_.benchmark || config_.render)) {
        throw std::invalid_argument("calibrate_nexto_ladder can't be set with benchmark or render.");
    }

    // A benchmark runner's config was already checked by the benchmark that made it.
    if (!benchmark_child_) {
        validate_rewards();
        validate_cpu_pinning();
    }

    // Converted before the benchmark returns below, so a bad value fails once here instead of in every candidate.
    episode_limits_ = {limit_ticks(config_.no_touch_timeout_seconds, "no_touch_timeout_seconds"),
                       limit_ticks(config_.episode_time_limit_seconds, "episode_time_limit_seconds")};
    if (!benchmark_child_ && episode_limits_.no_touch_ticks > 0 && episode_limits_.time_limit_ticks > 0 &&
        episode_limits_.no_touch_ticks > episode_limits_.time_limit_ticks) {
        warn("no_touch_timeout_seconds is longer than episode_time_limit_seconds, so it never ends an episode.",
             "No-touch timeout unused");
    }

    // Resolve the automatic env and thread counts once, so everything after sees the actual values.
    // Nothing is pinned yet, so the allowed CPUs are the whole process's.
    allowed_cpus_ = util::allowed_cpus().size();
    if (!config_.render) {
        config_.num_envs = config_.total_envs();
        // Unpinned, one CPU is left for the main thread, which is slower when it has to share every CPU with workers.
        if (config_.num_threads == 0) {
            config_.num_threads = config_.worker_cpus.empty() ? std::max<size_t>(1, allowed_cpus_ - 1) : config_.worker_cpus.size();
        }

        // Evaluation rates every even layout trained on, unless the team sizes are set.
        EvalConfig& eval = config_.eval_cfg;
        if (eval.enabled() && eval.team_sizes.empty()) {
            for (const MatchDistribution& dist : config_.match_distributions) {
                if (dist.weight > 0.0f && dist.blue_players == dist.orange_players) {
                    eval.team_sizes.push_back(dist.blue_players);
                }
            }
            if (eval.team_sizes.empty()) {
                throw std::invalid_argument("Evaluation needs even teams to rate. Set eval_cfg.team_sizes, or turn "
                                            "evaluation off with eval_cfg.steps_per_eval = 0.");
            }
            std::sort(eval.team_sizes.begin(), eval.team_sizes.end());
            eval.team_sizes.erase(std::unique(eval.team_sizes.begin(), eval.team_sizes.end()), eval.team_sizes.end());
        }
    }

    // The benchmark builds a runner for each config it measures instead. A ladder calibration only needs the threads.
    if ((config_.benchmark && !benchmark_child_) || config_.calibrate_nexto_ladder) {
        return;
    }

    setup_dimensions_and_buffers();
}

PPORunner::~PPORunner() = default;

int64_t PPORunner::load_latest_checkpoint(const std::string& dir) {
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
        return 0;
    }

    fs::path latest_dir;
    int64_t max_steps = -1;

    // Scan for checkpoint directories named <bot name>_<steps> that contain a model.
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_directory() || !fs::exists(entry.path() / MODEL_FILE)) continue;

        const std::string name = entry.path().filename().string();
        const size_t delim_pos = name.find_last_of('_');
        if (delim_pos == std::string::npos || name.substr(0, delim_pos) != config_.bot_name) continue;

        // The whole suffix must be the step count, so in-progress `<bot name>_<steps>.tmp` saves are skipped.
        int64_t steps = 0;
        const char* first = name.data() + delim_pos + 1;
        const char* last = name.data() + name.size();
        const auto [ptr, ec] = std::from_chars(first, last, steps);
        if (ec != std::errc{} || ptr != last || first == last) continue;

        if (steps > max_steps) {
            max_steps = steps;
            latest_dir = entry.path();
        }
    }

    if (max_steps < 0) {
        std::cout << "No valid checkpoints found for bot '" << config_.bot_name << "'. Starting fresh training.\n";
        return 0;
    }

    const std::string model_path = (latest_dir / MODEL_FILE).string();
    const ModelConfig current = model_config();

    // Checkpoints saved before models stored their config only get the weight check below.
    const std::optional<ModelConfig> saved = read_model_config(model_path);
    if (saved && (saved->network != current.network || saved->max_players_per_team != current.max_players_per_team)) {
        throw std::runtime_error("Could not resume from the latest checkpoint, " + model_path + ". It was trained with\n  " +
                                 describe_model_config(*saved) + "\nbut the config gives\n  " +
                                 describe_model_config(current) + "\nA changed obs_size or action_size means the obs "
                                 "builder or action parser changed. Change bot_name or move the checkpoint out of " +
                                 dir + " to start fresh.");
    }
    if (saved && saved->ticks_per_step != current.ticks_per_step) {
        warn("The latest checkpoint was trained with ticks_per_step " + std::to_string(saved->ticks_per_step) +
             ", but the config gives " + std::to_string(current.ticks_per_step) + ". Training continues with " +
             std::to_string(current.ticks_per_step) + ", and checkpoints from now on store it.",
             "ticks_per_step changed");
    }

    try {
        actor_critic_->load_weights(model_path);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Could not resume from the latest checkpoint: ") + e.what() +
                                 ". A changed network config (ac_cfg layers or layer norm), max_players_per_team or "
                                 "obs builder makes old checkpoints unusable. Change bot_name or move the checkpoint "
                                 "out of " + dir + " to start fresh.");
    }
    std::cout << "Latest Model Found: " << latest_dir.string() << " (Lifetime Steps: " << max_steps << ")\n";

    const fs::path optimizer_path = latest_dir / OPTIMIZER_FILE;
    if (!fs::exists(optimizer_path)) {
        std::cout << "No optimizer state found. Starting with a fresh optimizer.\n";
    } else if (trainer_->load_optimizer(optimizer_path.string())) {
        std::cout << "Optimizer state restored.\n";
    } else {
        warn("The checkpoint's optimizer state could not be used, see above. Starting with a fresh optimizer.",
             "Fresh optimizer");
    }
    return max_steps;
}

void PPORunner::validate_rewards() {
    // Only layouts that get envs matter.
    size_t largest_team = 0;
    bool has_empty_team = false;
    for (const auto& dist : config_.match_distributions) {
        if (dist.weight <= 0.0f) continue;
        largest_team = std::max({largest_team, dist.blue_players, dist.orange_players});
        has_empty_team |= dist.blue_players == 0 || dist.orange_players == 0;
    }

    for (const std::string& warning : reward::validate_reward_entries(config_.reward_entries, largest_team, has_empty_team)) {
        warn(warning, "Reward config");
    }
}

void PPORunner::validate_cpu_pinning() {
    // Runs before any thread is pinned, so the allowed CPUs are the whole process's, e.g. as limited by `taskset`.
    util::validate_cpus(config_.main_thread_cpus, "main_thread_cpus");
    util::validate_cpus(config_.worker_cpus, "worker_cpus");

    for (const int cpu : config_.main_thread_cpus) {
        if (std::find(config_.worker_cpus.begin(), config_.worker_cpus.end(), cpu) != config_.worker_cpus.end()) {
            warn("main_thread_cpus and worker_cpus share CPU " + std::to_string(cpu)
                 + ", so the env workers can take turns with the main thread on it.", "Main thread shares CPU " + std::to_string(cpu));
            break;
        }
    }
}

void PPORunner::setup_dimensions_and_buffers() {

    if (config_.render) {
        config_.num_envs = 1;
        config_.num_threads = 1;
        config_.num_minibatches = 1;
    }

    vec_env_ = std::make_unique<env::VecEnv>(
        config_.total_envs(),
        config_.match_distributions,
        config_.setter_distributions,
        config_.reward_entries,
        config_.metric_entries,
        config_.num_threads, 
        config_.ticks_per_step, 
        episode_limits_,
        config_.max_players_per_team,
        config_.render,
        config_.pfsp_cfg,
        config_.worker_cpus,
        config_.render_external_bots
    );

    total_agents_ = vec_env_->get_total_agents();
    learner_agents_ = vec_env_->get_learner_agents();
    obs_size_ = vec_env_->get_single_obs_size();
    const size_t action_space_size = vec_env_->get_action_space_size();

    if (config_.render) {
        config_.target_steps_per_update = learner_agents_;
    }

    // Every minibatch must be the same size for the captured training graph, so the buffer grows until the
    // minibatches divide its steps. That's usually a step or two, but some minibatch counts need many more.
    buffer_size_ = (config_.target_steps_per_update + learner_agents_ - 1) / learner_agents_;
    const size_t target_buffer_size = buffer_size_;
    while ((buffer_size_ * learner_agents_) % config_.num_minibatches != 0) {
        buffer_size_++;
    }
    if (!benchmark_child_ && buffer_size_ * 20 > target_buffer_size * 21) {
        // The minibatch count closest to the configured one that needs no growth. 1 always qualifies.
        const size_t target_steps = target_buffer_size * learner_agents_;
        const size_t requested = config_.num_minibatches;
        const auto distance = [requested](size_t m) { return m > requested ? m - requested : requested - m; };
        size_t suggested = 1;
        for (size_t m = 2; m <= 2 * requested; m++) {
            if (target_steps % m == 0 && distance(m) < distance(suggested)) suggested = m;
        }
        std::ostringstream growth;
        growth << std::fixed << std::setprecision(1)
               << 100.0 * static_cast<double>(buffer_size_ - target_buffer_size) / static_cast<double>(target_buffer_size);
        warn("num_minibatches = " + std::to_string(config_.num_minibatches) + " doesn't divide the steps per update "
             "evenly, so each agent's steps per update grew from " + std::to_string(target_buffer_size) + " to " +
             std::to_string(buffer_size_) + " (+" + growth.str() + "% steps, time and memory per update). "
             "num_minibatches = " + std::to_string(suggested) + " would keep " + std::to_string(target_buffer_size) + ".",
             "Steps per update +" + growth.str() + "%");
    }

    // Initialize staging buffers for deferred truncation critic forward pass.
    // Reserve about 2× the rows a typical update stages; vectors keep capacity after clear(), so a larger update grows them once.
    deferred_idx_.reserve(2 * learner_agents_);
    deferred_obs_.reserve(2 * learner_agents_ * obs_size_);

    total_steps_per_update_ = static_cast<int64_t>(buffer_size_ * learner_agents_);
    config_.ppo_cfg.mini_batch_size = total_steps_per_update_ / config_.num_minibatches;

    if (!benchmark_child_ && !config_.render) {
        std::cout << "Envs: " << config_.num_envs << " (" << learner_agents_ << " learner agents, " << buffer_size_
                  << " steps each per update), worker threads: " << config_.num_threads << "\n";
    }

    config_.ac_cfg.obs_size = static_cast<int64_t>(obs_size_);
    config_.ac_cfg.action_size = static_cast<int64_t>(action_space_size);
    
    actor_critic_ = ActorCritic(config_.ac_cfg);
    actor_critic_->to(device_);

    trainer_ = std::make_unique<PPOTrainer>(config_.ppo_cfg, actor_critic_, device_, !benchmark_child_);

    // Opponent slots only run inference, so their weights never need gradients.
    for (size_t slot = 0; slot < vec_env_->get_num_opponent_slots(); slot++) {
        ActorCritic opponent(config_.ac_cfg);
        opponent->to(device_);
        for (auto& param : opponent->parameters()) {
            param.requires_grad_(false);
        }
        opponent_slots_.push_back(opponent);
        opponent_streams_.push_back(at::cuda::getStreamFromPool());
    }
    if (!opponent_slots_.empty()) {
        opponent_pool_ = std::make_unique<OpponentPool>(config_.pfsp_cfg, model_config());
    }

    // Setup gauntlet for evaluation.
    if (config_.eval_cfg.enabled() && !config_.render && !benchmark_child_) {
        gauntlet_ = std::make_unique<Gauntlet>(config_.eval_cfg, model_config(), config_.eval_cfg.team_sizes,
                                               config_.num_threads, config_.worker_cpus, device_);
    }

    
    buffer_ = std::make_unique<RolloutBuffer>(
        buffer_size_, learner_agents_, obs_size_, action_space_size, device_
    );

    // Actions cover every row, since VecEnv reads each agent's action from its row.
    // The step tensors only hold learner rows.
    const int64_t t_agents = static_cast<int64_t>(total_agents_);
    const int64_t l_agents = static_cast<int64_t>(learner_agents_);
    const auto cpu_int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    actions_cpu_ = torch::empty({t_agents}, cpu_int_opts);

    const auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    step_obs_gpu_ = torch::empty({l_agents, static_cast<int64_t>(obs_size_)}, float_opts);
    step_masks_gpu_ = torch::empty({l_agents, static_cast<int64_t>(action_space_size)}, float_opts);
    step_rewards_gpu_ = torch::empty({l_agents}, float_opts);
    step_term_gpu_ = torch::empty({l_agents}, float_opts);
    step_trunc_gpu_ = torch::empty({l_agents}, float_opts);
    step_dones_gpu_ = torch::empty({l_agents}, float_opts);

    const int64_t o_agents = t_agents - l_agents;
    opp_obs_gpu_ = torch::empty({o_agents, static_cast<int64_t>(obs_size_)}, float_opts);
    opp_masks_gpu_ = torch::empty({o_agents, static_cast<int64_t>(action_space_size)}, float_opts);
}

void PPORunner::setup_rollout_groups(const torch::Tensor& obs, const torch::Tensor& masks) {
    rollout_groups_.clear();
    for (const env::EnvGroup& env_group : vec_env_->get_groups()) {
        auto group = std::make_unique<RolloutGroup>();
        const auto begin = static_cast<int64_t>(env_group.learner_begin);
        const auto count = static_cast<int64_t>(env_group.learner_count());
        group->learner_begin = begin;
        group->learner_count = count;

        group->obs_src = obs.narrow(0, begin, count);
        group->masks_src = masks.narrow(0, begin, count);
        group->actions_cpu = actions_cpu_.narrow(0, begin, count);
        group->obs_gpu = step_obs_gpu_.narrow(0, begin, count);
        group->masks_gpu = step_masks_gpu_.narrow(0, begin, count);
        group->rewards_gpu = step_rewards_gpu_.narrow(0, begin, count);
        group->term_gpu = step_term_gpu_.narrow(0, begin, count);
        group->trunc_gpu = step_trunc_gpu_.narrow(0, begin, count);
        group->dones_gpu = step_dones_gpu_.narrow(0, begin, count);
        group->buffer_step_gpu = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64).device(device_));

        for (const env::RowRange& rows : env_group.opponent_slots) {
            const auto opp_begin = static_cast<int64_t>(rows.begin);
            const auto opp_count = static_cast<int64_t>(rows.count);
            const int64_t staging_begin = opp_begin - static_cast<int64_t>(learner_agents_);
            group->opponents.push_back({
                obs.narrow(0, opp_begin, opp_count),
                masks.narrow(0, opp_begin, opp_count),
                opp_obs_gpu_.narrow(0, staging_begin, opp_count),
                opp_masks_gpu_.narrow(0, staging_begin, opp_count),
                actions_cpu_.narrow(0, opp_begin, opp_count)
            });
        }

        rollout_groups_.push_back(std::move(group));
    }
}

void PPORunner::load_opponent_slot(size_t slot, const std::vector<torch::Tensor>& params) {
    const std::vector<torch::Tensor> slot_params = opponent_slots_.at(slot)->parameters();
    TORCH_CHECK(slot_params.size() == params.size(), "Opponent weights don't match the network config.");

    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < params.size(); i++) {
        if (!params[i].defined()) continue;
        TORCH_CHECK(slot_params[i].sizes() == params[i].sizes(), "Opponent weights don't match the network config.");
        slot_params[i].copy_(params[i], true);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PPORunner::inference_graph_logic(RolloutGroup& group) {
    // The opponent slots fork off here. An opponent forward is a chain of small kernels that leaves the GPU mostly
    // idle, so running each slot on its own stream overlaps it with the learner's forward instead of queuing after it.
    const at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    at::cuda::CUDAEvent fork;
    fork.record(main_stream);

    group.obs_gpu.copy_(group.obs_src, true);
    group.masks_gpu.copy_(group.masks_src, true);

    auto [actions, log_probs, values] = actor_critic_->get_action_and_value(group.obs_gpu, group.masks_gpu);

    // Non-blocking, since synchronizing isn't allowed during capture. `infer_group` syncs after replay.
    group.actions_cpu.copy_(actions.to(torch::kInt32), true);

    for (size_t slot = 0; slot < group.opponents.size(); slot++) {
        RolloutGroup::OpponentRows& opp = group.opponents[slot];
        if (opp.obs_src.size(0) == 0) continue;

        const at::cuda::CUDAStream slot_stream = opponent_streams_[slot];
        fork.block(slot_stream);
        {
            at::cuda::CUDAStreamGuard stream_guard(slot_stream);
            opp.obs_gpu.copy_(opp.obs_src, true);
            opp.masks_gpu.copy_(opp.masks_src, true);
            const torch::Tensor opp_actions = opponent_slots_[slot]->sample_action(opp.obs_gpu, opp.masks_gpu);
            opp.actions_cpu.copy_(opp_actions.to(torch::kInt32), true);
        }

        // Join back. The learner's work is already queued ahead of this wait, so it doesn't wait on the slot.
        // A capture must rejoin every stream it forked before it ends.
        at::cuda::CUDAEvent join;
        join.record(slot_stream);
        join.block(main_stream);
    }

    return { actions, log_probs, values.squeeze(-1) };
}

void PPORunner::infer_group(RolloutGroup& group, const torch::Tensor& obs, const torch::Tensor& masks) {
    if (!group.captured) {
        // Graph capture has to happen on a private stream, after any work already queued on the current stream.
        at::cuda::CUDAStream capture_stream = at::cuda::getStreamFromPool();
        {
            at::cuda::CUDAEvent ready;
            ready.record(at::cuda::getCurrentCUDAStream());
            ready.block(capture_stream);
        }
        {
            at::cuda::CUDAStreamGuard stream_guard(capture_stream);

            // Warmup runs lazy initialization (e.g. cuBLAS workspaces) outside of capture. It only consumes RNG.
            for (int i = 0; i < 3; i++) {
                (void)inference_graph_logic(group);
            }
            capture_stream.synchronize();

            group.graph.capture_begin();
            std::tie(group.actions_gpu, group.log_probs_gpu, group.values_gpu) = inference_graph_logic(group);
            group.graph.capture_end();
        }
        group.captured = true;
    }

    // The graph baked in the source rows' addresses.
    TORCH_CHECK(obs.data_ptr() == group.obs_src.data_ptr() && masks.data_ptr() == group.masks_src.data_ptr(),
                "Rollout inference graph inputs must be the group's rows of VecEnv's persistent obs and mask buffers.");

    group.graph.replay();
    at::cuda::getCurrentCUDAStream().synchronize();
}

void PPORunner::insert_graph_logic(RolloutGroup& group) {
    group.rewards_gpu.copy_(group.rewards_src, true);
    group.term_gpu.copy_(group.term_src, true);
    group.trunc_gpu.copy_(group.trunc_src, true);
    torch::max_out(group.dones_gpu, group.term_gpu, group.trunc_gpu);

    buffer_->insert(
        group.buffer_step_gpu, group.learner_begin,
        group.obs_gpu, group.actions_gpu, group.masks_gpu,
        group.rewards_gpu, group.dones_gpu, group.log_probs_gpu, group.values_gpu
    );
    group.buffer_step_gpu.add_(1);
}

void PPORunner::finish_group_step(RolloutGroup& group, size_t step, const env::BatchedStepResult& step_res) {
    // The insert graph writes the step on the device, so the host's step has to be the one it writes next.
    if (step != group.buffer_step || step >= buffer_size_) {
        throw std::logic_error("Rollout step " + std::to_string(step) + " doesn't match the buffer step "
                               + std::to_string(group.buffer_step) + " the insert graph writes next.");
    }

    // Store truncated states for deferred bootstrapping.
    const float* truncated = step_res.truncated.data_ptr<float>();
    const float* terminal_obs = step_res.terminal_observations.data_ptr<float>();
    const int64_t obs_size = step_res.terminal_observations.size(1);

    for (int64_t i = 0; i < group.learner_count; i++) {
        if (truncated[i] == 0.0f) continue;
        deferred_idx_.push_back(static_cast<int64_t>(step) * static_cast<int64_t>(learner_agents_) + group.learner_begin + i);
        const float* row = terminal_obs + i * obs_size;
        deferred_obs_.insert(deferred_obs_.end(), row, row + obs_size);
    }

    if (!group.insert_captured) {
        group.rewards_src = step_res.rewards;
        group.term_src = step_res.terminated;
        group.trunc_src = step_res.truncated;

        // This first step runs eagerly, which also warms up the ops before the capture.
        insert_graph_logic(group);

        // Capture on a private stream, after the eager step. Capturing doesn't run the work again.
        at::cuda::CUDAStream capture_stream = at::cuda::getStreamFromPool();
        {
            at::cuda::CUDAEvent ready;
            ready.record(at::cuda::getCurrentCUDAStream());
            ready.block(capture_stream);
        }
        {
            at::cuda::CUDAStreamGuard stream_guard(capture_stream);
            group.insert_graph.capture_begin();
            insert_graph_logic(group);
            group.insert_graph.capture_end();
        }
        group.insert_captured = true;
    } else {
        // The graph baked in the result rows' addresses.
        TORCH_CHECK(step_res.rewards.data_ptr() == group.rewards_src.data_ptr()
                    && step_res.terminated.data_ptr() == group.term_src.data_ptr()
                    && step_res.truncated.data_ptr() == group.trunc_src.data_ptr(),
                    "Rollout insert graph inputs must be the group's rows of VecEnv's persistent result buffers.");
        group.insert_graph.replay();
    }
    group.buffer_step++;
}

double PPORunner::collect_rollout() {
    // Pipelined rollout over two env groups: while one group's physics runs on the workers, the main thread
    // uploads the other group's results and runs its inference. Each env still acts on its own latest obs
    // before it advances, so there is no action delay; only the order of steps across groups changes.
    RolloutGroup& group_a = *rollout_groups_[0];
    RolloutGroup& group_b = *rollout_groups_[1];
    int* const actions_ptr = actions_cpu_.data_ptr<int>();
    const size_t num_steps = buffer_size_;

    // Time the main thread spends blocked on the env workers. A large share of the rollout means physics is the
    // limit, and a share near 0 means the main thread's own work is.
    double physics_wait_seconds = 0.0;
    const auto wait_for_physics = [&] {
        const auto wait_start = std::chrono::high_resolution_clock::now();
        env::BatchedStepResult res = vec_env_->wait();
        physics_wait_seconds += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wait_start).count();
        return res;
    };

    for (const auto& group : rollout_groups_) {
        group->buffer_step = 0;
        group->buffer_step_gpu.zero_();
    }

    infer_group(group_a, group_a.obs_src, group_a.masks_src);
    vec_env_->step_async(0, actions_ptr);
    infer_group(group_b, group_b.obs_src, group_b.masks_src);

    for (size_t t = 0; t < num_steps; ++t) {
        const bool has_next = t + 1 < num_steps;

        const auto res_a = wait_for_physics();
        vec_env_->step_async(1, actions_ptr);
        finish_group_step(group_a, t, res_a);
        if (has_next) infer_group(group_a, res_a.observations, res_a.action_masks);

        const auto res_b = wait_for_physics();
        if (has_next) vec_env_->step_async(0, actions_ptr);
        finish_group_step(group_b, t, res_b);
        if (has_next) infer_group(group_b, res_b.observations, res_b.action_masks);
    }

    // Every group must have written every step exactly once, on the device too. A step the graphs skipped would
    // otherwise leave the last rollout's data in its place without an error.
    for (const auto& group : rollout_groups_) {
        const int64_t device_steps = group->buffer_step_gpu.item<int64_t>();
        if (group->buffer_step != num_steps || device_steps != static_cast<int64_t>(num_steps)) {
            throw std::logic_error("The rollout buffer got " + std::to_string(device_steps) + " steps on the GPU and "
                                   + std::to_string(group->buffer_step) + " on the CPU from an env group, but needs "
                                   + std::to_string(num_steps) + ".");
        }
    }

    // Deferred truncations forward pass for values.
    const int64_t n = static_cast<int64_t>(deferred_idx_.size());
    if (n > 0) {
        torch::Tensor deferred_idx_gpu = torch::from_blob(deferred_idx_.data(), n, torch::kInt64).to(device_);
        torch::Tensor deferred_obs_gpu = torch::from_blob(deferred_obs_.data(), {n, static_cast<int64_t>(obs_size_)}, torch::kFloat32).to(device_);
        torch::Tensor truncated_values;
        {
            torch::NoGradGuard no_grad;
            truncated_values = actor_critic_->forward_critic(deferred_obs_gpu).squeeze(-1);
        }
        
        buffer_->rewards_.view({-1}).index_add_(0, deferred_idx_gpu, truncated_values * config_.ppo_cfg.gae_gamma);
    }
    deferred_idx_.clear();
    deferred_obs_.clear();

    return physics_wait_seconds;
}

void PPORunner::warn(const std::string& message, const std::string& reminder) {
    // cout is fully buffered when redirected, so flush it first to keep the warning next to what it's about.
    std::cout << std::flush;
    std::cerr << "[WARN] " << message << "\n";
    warnings_.push_back({message, reminder, current_update_});
}

void PPORunner::print_warning_reminders() const {
    if (warnings_.empty()) return;

    std::vector<std::string> shown;
    std::cout << " | Warnings:    ";
    for (const RunWarning& warning : warnings_) {
        if (std::find(shown.begin(), shown.end(), warning.reminder) != shown.end()) continue;
        std::cout << (shown.empty() ? "" : ", ") << warning.reminder << " ("
                  << (warning.update == 0 ? "startup" : "update " + std::to_string(warning.update)) << ")";
        shown.push_back(warning.reminder);
    }
    std::cout << "\n";
}

void PPORunner::print_warning_summary() const {
    if (warnings_.empty()) return;

    std::cout << std::flush;
    std::cerr << "\nWarnings from this run:\n";
    for (const RunWarning& warning : warnings_) {
        std::cerr << " [" << (warning.update == 0 ? "startup" : "update " + std::to_string(warning.update)) << "] "
                  << warning.message << "\n";
    }
    std::cerr << "\n";
}

void PPORunner::check_rollout_bottleneck(double physics_wait) {
    // About 20 updates is long enough to ignore single slow updates and short swings in the bot's play.
    constexpr size_t WINDOW = 20;
    // Phys Wait above this means physics is the limit, below the other the main thread is.
    constexpr double PHYSICS_BOUND = 0.30;
    constexpr double MAIN_BOUND = 0.10;
    // CPUs without a worker before physics-bound warns. A single one is normal: the main thread's spare hyperthread.
    constexpr size_t MIN_IDLE_CPUS = 2;

    if (recent_physics_wait_.size() < WINDOW) {
        recent_physics_wait_.push_back(physics_wait);
    } else {
        recent_physics_wait_[next_physics_wait_] = physics_wait;
    }
    next_physics_wait_ = (next_physics_wait_ + 1) % WINDOW;
    if (recent_physics_wait_.size() < WINDOW) return;

    std::vector<double> sorted = recent_physics_wait_;
    std::nth_element(sorted.begin(), sorted.begin() + WINDOW / 2, sorted.end());
    const double median_wait = sorted[WINDOW / 2];
    const int wait_percent = static_cast<int>(std::lround(100.0 * median_wait));

    // CPUs with no worker on them: the main thread takes its pinned CPUs, or one CPU when unpinned. Pinned workers
    // can't use CPUs outside worker_cpus, and too few threads leave some of their CPUs empty.
    const std::vector<int>& main_cpus = config_.main_thread_cpus;
    const std::vector<int>& worker_cpus = config_.worker_cpus;
    const size_t main_taken = main_cpus.empty() ? 1 : main_cpus.size();
    size_t idle_cpus = 0;
    if (worker_cpus.empty()) {
        idle_cpus = allowed_cpus_ > main_taken + config_.num_threads ? allowed_cpus_ - main_taken - config_.num_threads : 0;
    } else {
        std::vector<int> used = worker_cpus;
        used.insert(used.end(), main_cpus.begin(), main_cpus.end());
        std::sort(used.begin(), used.end());
        used.erase(std::unique(used.begin(), used.end()), used.end());
        idle_cpus = (worker_cpus.size() > config_.num_threads ? worker_cpus.size() - config_.num_threads : 0)
                  + (allowed_cpus_ > used.size() ? allowed_cpus_ - used.size() : 0);
    }

    if (!warned_physics_bound_ && median_wait >= PHYSICS_BOUND && idle_cpus >= MIN_IDLE_CPUS) {
        warned_physics_bound_ = true;
        warn("Physics was the limit over the last " + std::to_string(WINDOW) + " updates (Phys Wait "
             + std::to_string(wait_percent) + "%), while about " + std::to_string(idle_cpus) + " CPUs had no env worker. "
             + "More worker threads should be faster: try num_threads = 0, or compare setups with benchmark mode.",
             "Physics-bound with ~" + std::to_string(idle_cpus) + " idle CPUs");
    }

    // Unpinned with a worker for every CPU, the main thread has no CPU to itself and waits its turn with the workers.
    const bool main_shares_cpus = main_cpus.empty() && worker_cpus.empty() && config_.num_threads >= allowed_cpus_;
    if (!warned_main_bound_ && median_wait <= MAIN_BOUND && main_shares_cpus && allowed_cpus_ >= 4) {
        warned_main_bound_ = true;
        warn("The main thread was the limit over the last " + std::to_string(WINDOW) + " updates (Phys Wait "
             + std::to_string(wait_percent) + "%), and shares every CPU with the " + std::to_string(config_.num_threads)
             + " env workers. Pinning it to a core of its own or a worker fewer may be faster: compare setups with "
             + "benchmark mode.", "Main-thread-bound, sharing every CPU");
    }
}

void PPORunner::compute_advantages() {
    // The full obs buffer holds every group's latest obs, and its first rows are the learners'.
    step_obs_gpu_.copy_(batched_obs_.narrow(0, 0, static_cast<int64_t>(learner_agents_)), true);
    torch::Tensor next_values;
    {
        torch::NoGradGuard no_grad;
        next_values = actor_critic_->forward_critic(step_obs_gpu_).squeeze(-1);
    }
    buffer_->compute_returns_and_advantages(next_values, config_.ppo_cfg.gae_gamma, config_.ppo_cfg.gae_lambda);
}

ModelConfig PPORunner::model_config() const {
    return {config_.ac_cfg, static_cast<int64_t>(config_.max_players_per_team), config_.ticks_per_step};
}

void PPORunner::save_checkpoint(const std::string& dir) const {
    const fs::path final_dir = fs::path(dir) / (config_.bot_name + "_" + std::to_string(global_step_));
    const fs::path tmp_dir = final_dir.string() + ".tmp";

    // Write into a temp dir and rename once complete so an interrupted save never looks like a valid checkpoint.
    fs::remove_all(tmp_dir);
    fs::create_directories(tmp_dir);
    save_model(actor_critic_, model_config(), (tmp_dir / MODEL_FILE).string());
    trainer_->save_optimizer((tmp_dir / OPTIMIZER_FILE).string());

    fs::remove_all(final_dir);
    fs::rename(tmp_dir, final_dir);

    // The pool saves its snapshots as they're added, so only the latest win records are left to write.
    if (opponent_pool_) {
        opponent_pool_->save_state();
    }
    std::cout << "Checkpoint saved to: " << final_dir.string() << "\n";
}

void PPORunner::write_log_window() {
    LogWindow& window = log_window_;
    const double steps = static_cast<double>(window.steps);
    const double updates = static_cast<double>(window.updates);

    // Rewards and metrics accumulated inside the envs since the last log.
    vec_env_->update_telemetry_breakdowns();
    // Components are the raw values, before the reward options. Magnitudes are the absolute values the agents received.
    double total_raw = 0.0;
    double total_magnitude = 0.0;
    for (const auto& [name, accumulation] : vec_env_->get_reward_breakdown()) {
        total_raw += accumulation.raw;
        total_magnitude += accumulation.magnitude;
        logger_->add_scalar("Reward_Components/" + name, global_step_, static_cast<float>(accumulation.raw / steps));
        logger_->add_scalar("Reward_Magnitude/" + name, global_step_, static_cast<float>(accumulation.magnitude / steps));
    }
    logger_->add_scalar("Reward/Mean_Step", global_step_, static_cast<float>(total_raw / steps));
    logger_->add_scalar("Reward/Mean_Abs_Step", global_step_, static_cast<float>(total_magnitude / steps));

    // Each reward's fraction of the total magnitude, which sums to 1. Skipped when no reward was received.
    if (total_magnitude > 0.0) {
        for (const auto& [name, accumulation] : vec_env_->get_reward_breakdown()) {
            logger_->add_scalar("Reward_Share/" + name, global_step_, static_cast<float>(accumulation.magnitude / total_magnitude));
        }
    }

    // Custom Metrics.
    for (const auto& [name, accumulation] : vec_env_->get_metric_breakdown()) {
        if (accumulation.count == 0) continue;
        logger_->add_scalar("Metrics/" + name, global_step_, static_cast<float>(accumulation.value / accumulation.count));
    }

    logger_->add_scalar("Episode/Goals", global_step_, static_cast<float>(window.goals));
    logger_->add_scalar("Episode/Truncations", global_step_,
                        static_cast<float>(window.no_touch_truncations + window.time_limit_truncations));
    logger_->add_scalar("Episode/Truncations_No_Touch", global_step_, static_cast<float>(window.no_touch_truncations));
    logger_->add_scalar("Episode/Truncations_Time_Limit", global_step_, static_cast<float>(window.time_limit_truncations));

    // PFSP totals over the window. Truncated episodes aren't counted.
    if (!opponent_slots_.empty()) {
        logger_->add_scalar("PFSP/Pool_Size", global_step_, static_cast<float>(opponent_pool_->size()));
        logger_->add_scalar("PFSP/Games", global_step_, static_cast<float>(window.pfsp_games));
        if (window.pfsp_games > 0) {
            logger_->add_scalar("PFSP/Win_Rate", global_step_,
                                static_cast<float>(window.pfsp_wins) / static_cast<float>(window.pfsp_games));
        }
    }

    // Trainer metrics are averaged over the window's updates, which all have the same number of steps.
    const auto mean_metric = [&](const std::string& name) {
        return static_cast<float>(window.train_metrics.at(name) / updates);
    };
    logger_->add_scalar("Performance/Total_SPS", global_step_, steps / window.update_seconds);
    logger_->add_scalar("Performance/Rollout_SPS", global_step_, steps / window.rollout_seconds);
    logger_->add_scalar("Performance/Rollout_Physics_Wait", global_step_, window.physics_wait_seconds / window.rollout_seconds);

    // What the envs cost to step since the last log. Touches and repredictions explain most of the step time's rise
    // as a bot learns to play the ball.
    const env::EnvCostCounts cost = vec_env_->collect_cost_counts();
    if (cost.env_steps > 0) {
        const auto env_steps = static_cast<double>(cost.env_steps);
        logger_->add_scalar("Performance/Env_Step_Microseconds", global_step_, 1e6 * cost.step_seconds / env_steps);
        logger_->add_scalar("Performance/Touch_Ticks_Per_Env_Step", global_step_, static_cast<double>(cost.touch_ticks) / env_steps);
        logger_->add_scalar("Performance/Repredictions_Per_Env_Step", global_step_, static_cast<double>(cost.repredictions) / env_steps);
    }
    logger_->add_scalar("Loss/Policy", global_step_, mean_metric("policy_loss"));
    logger_->add_scalar("Loss/Value", global_step_, mean_metric("value_loss"));
    logger_->add_scalar("Loss/Entropy", global_step_, mean_metric("entropy"));
    logger_->add_scalar("Policy/Normalized_Entropy", global_step_, mean_metric("normalized_entropy"));
    logger_->add_scalar("Policy/Approx_KL", global_step_, mean_metric("approx_kl"));
    logger_->add_scalar("Policy/Clip_Fraction", global_step_, mean_metric("clip_fraction"));
    logger_->add_scalar("Value/Explained_Variance", global_step_, mean_metric("explained_variance"));

    window = {};
}

void PPORunner::prepare_rollouts() {
    if (vec_env_->get_groups().size() < 2) {
        throw std::invalid_argument("Training needs at least 2 envs, since the rollout is pipelined over two env groups.");
    }

    const auto reset_res = vec_env_->reset();
    batched_obs_ = reset_res.observations;
    setup_rollout_groups(reset_res.observations, reset_res.action_masks);
}

void PPORunner::run_training(int num_updates, const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir) {

    fs::create_directories("logs");
    const std::string log_file = "logs/" + config_.bot_name + ".tfevents";
    TensorBoardLoggerOptions logger_options{};
    logger_options.resume_ = true;
    logger_ = std::make_unique<TensorBoardLogger>(log_file.c_str(), logger_options);

    // The main thread runs inference and the step bookkeeping. Pinned only here, after the env workers and CUDA have
    // started their threads, since threads it starts from now on inherit its CPUs.
    if (!config_.main_thread_cpus.empty() && !util::pin_thread(pthread_self(), config_.main_thread_cpus)) {
        warn("Could not pin the main thread to main_thread_cpus. It runs unpinned.", "Main thread unpinned");
    }

    prepare_rollouts();

    // Seed an empty pool with the starting learner, so there is always a version to play.
    const int64_t snapshot_interval = config_.pfsp_cfg.snapshot_interval;
    if (opponent_pool_) {
        if (opponent_pool_->empty()) {
            opponent_pool_->add(global_step_, actor_critic_);
        }
        next_snapshot_step_ = (global_step_ / snapshot_interval + 1) * snapshot_interval;
    }

    // Periodic checkpoints land on multiples of the interval, so they stay on schedule across resumes.
    const int64_t checkpoint_interval = config_.steps_per_checkpoint;
    if (checkpoint_interval > 0) {
        next_checkpoint_step_ = (global_step_ / checkpoint_interval + 1) * checkpoint_interval;
    }

    // Evaluations land on multiples of the interval, like checkpoints. A run without anchors gets its first one now.
    const int64_t steps_per_eval = config_.eval_cfg.steps_per_eval;
    if (gauntlet_) {
        if (gauntlet_->empty()) {
            run_evaluation();
        }
        next_eval_step_ = (global_step_ / steps_per_eval + 1) * steps_per_eval;
    }

    // TensorBoard logs land on multiples of the interval like checkpoints. 0 logs every update.
    const int64_t log_interval = config_.steps_per_log > 0 ? config_.steps_per_log : total_steps_per_update_;
    next_log_step_ = (global_step_ / log_interval + 1) * log_interval;

    for (int update = 1; update <= num_updates; ++update) {
        current_update_ = update;

        if (stop_flag) {
            std::cout << "\nTraining interrupted by user. Stopping...\n";
            break;
        }

        const auto t_start = std::chrono::high_resolution_clock::now();
        const auto rollout_start = std::chrono::high_resolution_clock::now();

        // This update's opponents: past versions sampled from the pool, or a copy of the current learner
        // until the PFSP start steps.
        std::vector<size_t> opponent_picks;
        std::vector<int64_t> opponent_versions; // Kept for the console, since a snapshot added later invalidates the picks.
        if (opponent_pool_ && global_step_ >= config_.pfsp_cfg.start_steps) {
            opponent_picks = opponent_pool_->sample(opponent_slots_.size());
            for (const size_t pick : opponent_picks) {
                opponent_versions.push_back((*opponent_pool_)[pick].steps);
            }
        }
        for (size_t slot = 0; slot < opponent_slots_.size(); slot++) {
            if (opponent_picks.empty()) {
                load_opponent_slot(slot, actor_critic_->parameters());
            } else {
                load_opponent_slot(slot, (*opponent_pool_)[opponent_picks[slot]].params);
            }
        }

        const double physics_wait_seconds = collect_rollout();

        // Successful rollout, increment global steps.
        global_step_ += total_steps_per_update_;

        // Episode counts are collected every update for the console, and summed into the log window.
        const env::EpisodeCounts episodes = vec_env_->collect_episode_counts();
        log_window_.goals += episodes.goals;
        log_window_.no_touch_truncations += episodes.no_touch_truncations;
        log_window_.time_limit_truncations += episodes.time_limit_truncations;

        // PFSP results against this update's opponent slots. Truncated episodes aren't counted.
        uint32_t pfsp_wins = 0;
        uint32_t pfsp_games = 0;
        const auto pfsp_outcomes = vec_env_->collect_pfsp_outcomes();
        for (size_t slot = 0; slot < pfsp_outcomes.size(); slot++) {
            const uint32_t games = pfsp_outcomes[slot].wins + pfsp_outcomes[slot].losses;
            pfsp_wins += pfsp_outcomes[slot].wins;
            pfsp_games += games;
            if (!opponent_picks.empty()) {
                opponent_pool_->record(opponent_picks[slot], pfsp_outcomes[slot].wins, pfsp_outcomes[slot].losses);
            }
        }
        log_window_.pfsp_wins += pfsp_wins;
        log_window_.pfsp_games += pfsp_games;

        const auto rollout_end = std::chrono::high_resolution_clock::now();

        // GAE & Optimize.
        const auto gae_start = std::chrono::high_resolution_clock::now();
        compute_advantages();
        const auto gae_end = std::chrono::high_resolution_clock::now();
        const auto metrics = trainer_->train_step(*buffer_);

        // Snapshot the freshly trained learner into the pool. Invalidates `opponent_picks`.
        if (opponent_pool_ && global_step_ >= next_snapshot_step_) {
            opponent_pool_->add(global_step_, actor_critic_);
            next_snapshot_step_ = (global_step_ / snapshot_interval + 1) * snapshot_interval;
        }

        const auto t_end = std::chrono::high_resolution_clock::now();
        
        const std::chrono::duration<double> update_time = t_end - t_start;
        const std::chrono::duration<double> rollout_time = rollout_end - rollout_start;
        const std::chrono::duration<double> gae_time = gae_end - gae_start;
        const double train_time = update_time.count() - rollout_time.count() - gae_time.count();

        const double total_sps = total_steps_per_update_ / update_time.count();
        const double rollout_sps = total_steps_per_update_ / rollout_time.count();
        const double physics_wait_percent = 100.0 * physics_wait_seconds / rollout_time.count();

        log_window_.steps += total_steps_per_update_;
        log_window_.updates++;
        log_window_.update_seconds += update_time.count();
        log_window_.rollout_seconds += rollout_time.count();
        log_window_.physics_wait_seconds += physics_wait_seconds;
        for (const auto& [name, value] : metrics) {
            log_window_.train_metrics[name] += value;
        }
        if (global_step_ >= next_log_step_) {
            write_log_window();
            next_log_step_ = (global_step_ / log_interval + 1) * log_interval;
        }

        std::cout << "Update: " << update 
                  << "\nLifetime Steps: " << global_step_
                  << "\n | Total SPS:   " << static_cast<int64_t>(total_sps)
                  << "\n |  | Rollout:  " << static_cast<int64_t>(rollout_sps)
                  << "\n | Time:        " << std::fixed << std::setprecision(4) << update_time.count() << "s"
                  << "\n |  | Rollout:  " << rollout_time.count() << "s"
                  << "\n |  | GAE:      " << gae_time.count() << "s"
                  << "\n |  | Train:    " << train_time << "s"
                  << "\n | Phys Wait:   " << std::setprecision(1) << physics_wait_percent << "% of rollout"
                  << "\n | Policy Loss: " << std::defaultfloat << std::setprecision(6) << metrics.at("policy_loss")
                  << "\n | Value Loss:  " << metrics.at("value_loss")
                  << "\n | Entropy:     " << metrics.at("entropy")
                  << "\n | Norm Ent:    " << metrics.at("normalized_entropy")
                  << "\n | Approx KL:   " << metrics.at("approx_kl")
                  << "\n | Clip Frac:   " << metrics.at("clip_fraction")
                  << "\n | Expl Var:    " << metrics.at("explained_variance")
                  << "\n | Episodes:    " << episodes.goals << " goals, " << episodes.no_touch_truncations << " no touch, "
                  << episodes.time_limit_truncations << " time limit\n";
        if (!pfsp_outcomes.empty()) {
            std::cout << " | PFSP Wins:   " << pfsp_wins << " / " << pfsp_games << " games";
            if (opponent_versions.empty()) {
                std::cout << " vs current learner";
            } else {
                std::cout << " vs";
                for (size_t slot = 0; slot < opponent_versions.size(); slot++) {
                    std::cout << (slot == 0 ? " " : ", ") << opponent_versions[slot];
                }
            }
            std::cout << " (pool: " << opponent_pool_->size() << ")\n";
        }

        check_rollout_bottleneck(physics_wait_seconds / rollout_time.count());
        print_warning_reminders();

        if (gauntlet_ && global_step_ >= next_eval_step_) {
            run_evaluation();
            next_eval_step_ = (global_step_ / steps_per_eval + 1) * steps_per_eval;
        }


        if (checkpoint_interval > 0 && global_step_ >= next_checkpoint_step_) {
            save_checkpoint(checkpoint_dir);
            last_checkpoint_step_ = global_step_;
            next_checkpoint_step_ = (global_step_ / checkpoint_interval + 1) * checkpoint_interval;
        }
    }

    // Log the partial window left by an interrupted or finished run.
    if (log_window_.updates > 0) {
        write_log_window();
    }

    // Skip the final save if the last update already wrote this checkpoint.
    if (global_step_ != last_checkpoint_step_) {
        save_checkpoint(checkpoint_dir);
    }

    // Last, so the warnings are the first thing seen after the run.
    print_warning_summary();
}

void PPORunner::run_render(const std::atomic<bool>& stop_flag) {
    auto reset_res = vec_env_->reset();
    torch::Tensor current_obs = reset_res.observations;
    torch::Tensor current_masks = reset_res.action_masks;

    std::cout << "Starting visualizer... Press Ctrl+C to stop.\n";

    while (!stop_flag) {
        step_obs_gpu_.copy_(current_obs, true);
        step_masks_gpu_.copy_(current_masks, true);

        const torch::Tensor actions_gpu = config_.render_best_action
            ? actor_critic_->best_action(step_obs_gpu_, step_masks_gpu_)
            : actor_critic_->sample_action(step_obs_gpu_, step_masks_gpu_);

        actions_cpu_.copy_(actions_gpu, false);
        
        // VecEnv blocks to maintain regular time scale.
        auto step_res = vec_env_->step(actions_cpu_.data_ptr<int>()); 

        current_obs = step_res.observations;
        current_masks = step_res.action_masks;
    }
}

void PPORunner::run_ladder_calibration(const std::string& checkpoint_dir) {
    const EvalConfig& eval = config_.eval_cfg;
    if (eval.nexto_calibration_bank_size == 0 || eval.nexto_calibration_rungs_above == 0) {
        throw std::invalid_argument("eval_cfg.nexto_calibration_bank_size and nexto_calibration_rungs_above must be positive.");
    }

    // The shipped ladder is for the default settings. A ladder for others goes where evaluation looks for one, so the
    // shipped file isn't overwritten, unless `nexto_ladder_path` was set to a file of the user's own.
    const EvalConfig defaults;
    const bool default_settings = eval.nexto_model_path == defaults.nexto_model_path && eval.pos_jitter == defaults.pos_jitter
                                  && eval.yaw_jitter == defaults.yaw_jitter && eval.game_cap_seconds == defaults.game_cap_seconds;
    const fs::path out = (default_settings || eval.nexto_ladder_path != defaults.nexto_ladder_path)
                             ? fs::path(eval.nexto_ladder_path)
                             : fs::path(checkpoint_dir) / NextoLadder::CALIBRATED_FILE;

    const NextoLadderSettings settings{eval.nexto_model_path, eval.pos_jitter, eval.yaw_jitter, eval.game_cap_seconds};
    NextoLadder::Calibration calibration;
    calibration.bank_size = eval.nexto_calibration_bank_size;
    calibration.bank_seed = eval.bank_seed;
    calibration.rungs_above = eval.nexto_calibration_rungs_above;
    calibration.num_threads = config_.num_threads;
    calibration.worker_cpus = config_.worker_cpus;
    calibration.max_players_per_team = config_.max_players_per_team;
    calibration.ticks_per_step = config_.ticks_per_step;
    calibration.virtual_draws = eval.virtual_draws;

    external::NextoOptions options;
    options.device = device_;
    const auto nexto = std::make_shared<external::NextoBot>(eval.nexto_model_path, options);
    std::cout << "Calibrating the Nexto ladder: " << calibration.bank_size << " kickoffs per team size and pairing, "
              << calibration.rungs_above << " rung(s) above each, for " << out.string() << ".\n";
    const std::string text = NextoLadder::calibrate(settings, calibration, nexto);
    if (!out.parent_path().empty()) fs::create_directories(out.parent_path());
    util::write_text_file(out, text);
    std::cout << "Wrote the Nexto ladder to " << out.string() << ". Evaluation uses it from the next start; an "
              << "evaluation ledger rated with another ladder is moved aside, and its Elo curve starts over.\n";
}

void PPORunner::run_evaluation() {
    const EvalResult result = gauntlet_->run(global_step_, actor_critic_);
    if (!result.played) {
        std::cout << " | Evaluation: the learner is now the first anchor, the reference at 0 Elo.\n";
        return;
    }

    for (const EvalSizeResult& size : result.sizes) {
        const std::string mode = std::to_string(size.team_size) + "v" + std::to_string(size.team_size);
        const double games = size.wins + size.losses + size.draws;
        logger_->add_scalar("Skill/" + mode + "_Elo", global_step_, size.elo);
        logger_->add_scalar("Skill/" + mode + "_Elo_Std_Error", global_step_, size.elo_std_error);
        // Against the snapshot anchors, which a run's first evaluation with Nexto doesn't play.
        if (games > 0) {
            logger_->add_scalar("Skill/" + mode + "_Score", global_step_, (size.wins + size.draws * 0.5f) / games);
            logger_->add_scalar("Skill/" + mode + "_Draw_Rate", global_step_, size.draws / games);
            logger_->add_scalar("Skill/" + mode + "_Mean_Game_Seconds", global_step_, size.mean_game_seconds);
        }
        if (size.nexto_beta) {
            logger_->add_scalar("Skill/" + mode + "_Nexto_Beta", global_step_, *size.nexto_beta);
            logger_->add_scalar("Skill/" + mode + "_Nexto_Score", global_step_, size.nexto_score);
        }
        if (size.nexto_gap_elo) {
            logger_->add_scalar("Skill/" + mode + "_Nexto_Gap", global_step_, *size.nexto_gap_elo);
            logger_->add_scalar("Skill/" + mode + "_Nexto_Gap_Std_Error", global_step_, size.nexto_gap_std_error);
        }
        std::cout << " | Eval " << mode << ":     " << std::fixed << std::setprecision(1) << size.elo << " +- "
                  << size.elo_std_error << " Elo";
        if (games > 0) {
            std::cout << " (W " << size.wins << " L " << size.losses << " D " << size.draws << ")";
        }
        if (size.nexto_gap_elo) {
            std::cout << ", Nexto gap " << *size.nexto_gap_elo << " +- " << size.nexto_gap_std_error;
        }
        std::cout << std::defaultfloat << "\n";
    }
    logger_->add_scalar("Skill/Active_Anchors", global_step_, static_cast<float>(result.anchors_played.size()));
    std::cout << " | Eval Played: ";
    for (size_t i = 0; i < result.anchors_played.size(); i++) {
        std::cout << (i == 0 ? "" : ", ") << result.anchors_played[i];
    }
    std::cout << (result.anchors_played.empty() ? "none" : "") << "\n";
    if (!result.nexto_played.empty()) {
        std::cout << " | Eval Nexto:  ";
        for (size_t i = 0; i < result.nexto_played.size(); i++) {
            const EvalNextoResult& n = result.nexto_played[i];
            std::cout << (i == 0 ? "" : ", ") << n.team_size << "v" << n.team_size << " beta " << n.beta << " (W "
                      << n.wins << " L " << n.losses << " D " << n.draws << ")";
        }
        std::cout << "\n";
    }
    logger_->add_scalar("Performance/Eval_Time", global_step_, static_cast<float>(result.seconds));
    std::cout << " | Eval Time:   " << std::fixed << std::setprecision(2) << result.seconds << "s"
              << (result.promoted ? ", the learner is now an anchor" : "") << std::defaultfloat << "\n";
}

void PPORunner::run(int num_updates, const std::atomic<bool>& stop_flag, const std::string& checkpoint_dir) {
    if (config_.benchmark) {
        run_benchmark(stop_flag);
        return;
    }
    if (config_.calibrate_nexto_ladder) {
        run_ladder_calibration(checkpoint_dir);
        return;
    }

    global_step_ = load_latest_checkpoint(checkpoint_dir);

    if (opponent_pool_) {
        opponent_pool_->open(fs::path(checkpoint_dir) / (config_.bot_name + POOL_DIR_SUFFIX), global_step_);
    }

    if (gauntlet_) {
        gauntlet_->open(fs::path(checkpoint_dir) / (config_.bot_name + EVAL_DIR_SUFFIX), global_step_);
    }

    if (config_.render) {
        run_render(stop_flag);
    } else {
        run_training(num_updates, stop_flag, checkpoint_dir);
    }
}

} // namespace buta_ppo::rl