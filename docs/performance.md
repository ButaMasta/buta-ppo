# Performance

## Benchmarking Your Hardware

The fastest thread count, CPU pinning, and env count depend on the CPU, the network size, the match layouts, and how the bot plays. A fresh bot spends most of its time in the air, which is cheap to simulate, so the main thread's inference is the limit. A bot that keeps touching the ball costs about twice as much per step, mostly in ball prediction, so physics becomes the limit. A config tuned on a fresh bot can lose 15% or more once the bot learns to play the ball.

Benchmark mode measures this for you instead of training:

```cpp
config.benchmark = true;
```

It asks for confirmation, then takes a few minutes (about 3 for the demo config). Nothing is loaded, saved, or logged. Each config is measured under two workloads:

- **Light:** a fresh policy with this config's network, mostly airborne.
- **Heavy:** every car is a scripted ball-chaser, with extra ball predictions to match a bot that keeps contact with the ball. It is the same on every machine, so it doesn't need a trained checkpoint.

The benchmark first compares CPU setups found from the CPU's layout: the current config, a few unpinned thread counts, and the main thread pinned to the fastest core. Then it tries a few env counts around the configured one (`num_envs`, or the one picked from `target_steps_per_agent`) for that setup. Each config runs real training updates, rollout and training both, since rollouts back to back without the training in between run at a different speed on some setups. Each row shows the total SPS under both workloads, the steps per agent, and `Worst`, the lower of the two SPS as a share of the best in its column. The recommended CPU setup is the one with the best `Worst`, or the best unpinned one when it is within 2 points, since pinned CPU ids don't carry over to other machines. The recommended env count is the configured one, unless an env count with steps per agent within about a third of it scores at least 5 points higher; the output says when one scored higher but not by enough. It is all printed ready to paste:

```
Recommended config:
    config.num_threads = 0;             // 19 workers on this machine.
    config.num_envs = 0;                // 500 envs from target_steps_per_agent = 100.
    config.main_thread_cpus = {};       // Unpinned.
    config.worker_cpus = {};            // Unpinned.
```

Values that the automatic settings already give are printed as `0`, so they keep fitting if the machine or `target_steps_per_update` changes, and empty CPU lists mean unpinned. Replace all four lines in your config, including clearing CPU lists from an earlier pinned setup.

Run the benchmark the way you train, with whatever usually runs alongside (a browser, a video, Discord) open, since those take CPU time from the workers.

Steps per agent are a learning choice, so the benchmark only moves away from yours for a clear gain, and the table shows what other env counts would gain. The training time and peak VRAM are printed too. They depend on `target_steps_per_update`, `num_minibatches`, and the network, not on the rows.

## Finding the Rollout Bottleneck

During the rollout, the main thread runs inference for one half of the envs while the worker threads step the physics of the other half. The console's `Phys Wait` line (and `Performance/Rollout_Physics_Wait` in TensorBoard, as a fraction) is the share of the rollout the main thread spent waiting for the workers:

- **High (about 25% and up):** physics is the limit. More worker threads help, up to the number of logical CPUs.
- **Low (about 10%):** the main thread's own work is the limit, so faster physics won't help much. On CPUs with spare cores, giving the main thread a core of its own can:

```cpp
// Logical CPU ids, as listed by `lscpu -e`. Empty (the default) leaves the threads unpinned.
config.main_thread_cpus = {0};                     // e.g. P-core 0 (CPUs 0 and 1 on hyperthreaded P-cores), CPU 1 left free.
config.worker_cpus = {2, 3, 4, /* ... */ 19};      // The workers share every other CPU.
config.num_threads = 0;                            // One worker per worker CPU.
```

On a 6 P-core + 8 E-core CPU this was about 5% faster, since the E-cores join the workers without competing with the main thread. On a 4-core CPU, where physics was the limit, reserving a core for the main thread was 6–10% slower, so measure before and after, or let [benchmark mode](#benchmarking-your-hardware) compare them. Startup fails if a listed CPU doesn't exist or is excluded by `taskset`, and warns if both lists share a CPU.

Phys Wait swings as the bot's play changes: a bot that keeps touching the ball makes physics the limit, and one that doesn't makes the main thread the limit. A setup from benchmark mode accounts for both. Training prints a `[WARN]`, once per run, when the median Phys Wait of the last 20 updates shows the CPU setup leaving speed unused:

- Physics is the limit (30% and up) while 2 or more CPUs have no worker thread.
- The main thread is the limit (10% and below) while it's unpinned and there are as many workers as CPUs, so it has no CPU of its own.
