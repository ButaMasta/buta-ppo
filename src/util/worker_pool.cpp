// buta-ppo/src/util/worker_pool.cpp
#include "worker_pool.hpp"
#include "thread_affinity.hpp"

#include <iostream>

namespace buta_ppo::util {

WorkerPool::WorkerPool(size_t num_threads, const std::vector<int>& cpus) {
    // Workers given CPUs all share them, so the OS still balances the workers across them.
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; i++) {
        workers_.emplace_back(&WorkerPool::worker_loop, this);
        if (!cpus.empty() && !pin_thread(workers_.back().native_handle(), cpus)) {
            std::cerr << "[WARN] Could not pin worker thread " << i << " to worker_cpus. It runs unpinned.\n";
        }
    }
}

WorkerPool::~WorkerPool() {
    {
        std::lock_guard<std::mutex> lock(start_mutex_);
        terminate_ = true;
    }
    cv_start_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void WorkerPool::worker_loop() {
    int local_batch_count = 0; // Track this worker's batch count.

    while (!terminate_) {
        // Wait for the dispatching thread.
        {
            std::unique_lock<std::mutex> lock(start_mutex_);
            cv_start_.wait(lock, [this, &local_batch_count] {
                return (batch_count_ > local_batch_count) || terminate_;
            });
        }

        if (terminate_) break;

        local_batch_count = batch_count_;

        // Claim indices until all are taken. Relaxed is enough: the batch_count_/pending_workers_ handshake
        // orders an index's writes by one worker before its next use by another.
        const Task& task = *task_;
        for (size_t i = next_index_.fetch_add(1, std::memory_order_relaxed); i < batch_end_;
             i = next_index_.fetch_add(1, std::memory_order_relaxed)) {
            task(i);
        }

        // If this was the last thread then the batch is complete.
        if (--pending_workers_ == 0) {
            std::lock_guard<std::mutex> lock(done_mutex_);
            cv_done_.notify_one();
        }
    }
}

void WorkerPool::dispatch(const Task& task, size_t begin, size_t end) {
    {
        std::lock_guard<std::mutex> lock(start_mutex_);
        task_ = &task;
        next_index_ = begin;
        batch_end_ = end;
        pending_workers_ = static_cast<int>(workers_.size());
        batch_count_++; // New batch work is ready.
    }
    cv_start_.notify_all();
}

void WorkerPool::wait() {
    {
        std::unique_lock<std::mutex> lock(done_mutex_);
        cv_done_.wait(lock, [this] { return pending_workers_ == 0; });
    }
    task_ = nullptr;
}

} // namespace buta_ppo::util
