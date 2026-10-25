// buta-ppo/src/util/worker_pool.hpp
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace buta_ppo::util {

/**
 * @brief A pool of worker threads that runs a task once for every index of a range, like each env of a batch.
 *
 * Workers claim indices one at a time from a shared counter until all are taken, so faster or earlier-woken threads
 * take on more of the work. One batch runs at a time: `dispatch` starts it and returns immediately, `wait` blocks
 * until it has finished.
 */
class WorkerPool {
public:
    using Task = std::function<void(size_t index)>;

    /**
     * @brief Starts the worker threads, which sleep until the first batch.
     *
     * @param num_threads The number of worker threads.
     * @param cpus The logical CPUs the workers share. Empty leaves them unpinned.
     */
    WorkerPool(size_t num_threads, const std::vector<int>& cpus = {});

    ~WorkerPool();

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    /**
     * @brief Wakes the workers to run `task` for every index in [begin, end). Returns immediately.
     *
     * The pool keeps a reference to `task`, so it must stay alive until `wait` returns. Building the tasks once
     * (e.g. as members) instead of per batch also avoids allocating on every batch.
     */
    void dispatch(const Task& task, size_t begin, size_t end);

    /**
     * @brief Blocks until the workers have finished the dispatched batch.
     */
    void wait();

    [[nodiscard]] size_t size() const { return workers_.size(); }

private:
    std::vector<std::thread> workers_;
    std::atomic<bool> terminate_{false};
    std::atomic<int> pending_workers_{0};

    // Identifier for what batch number is expected of all workers.
    std::atomic<int> batch_count_{0};

    // Index of the next item to be claimed by a worker in the current batch. Set to the batch's first index.
    std::atomic<size_t> next_index_{0};
    // One past the last index and the task of the current batch. Written by the dispatching thread before each batch.
    size_t batch_end_ = 0;
    const Task* task_ = nullptr;

    std::mutex start_mutex_;
    std::condition_variable cv_start_;

    std::mutex done_mutex_;
    std::condition_variable cv_done_;

    /**
     * @brief The loop all workers run: sleep until a batch is dispatched, claim and run its indices, repeat.
     */
    void worker_loop();
};

} // namespace buta_ppo::util
