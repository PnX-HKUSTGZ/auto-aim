#ifndef ARMOR_DETECTOR_THREAD_POOL_HPP_
#define ARMOR_DETECTOR_THREAD_POOL_HPP_

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

namespace rm_auto_aim {

// 线程池类，用于多线程优化BA
class ThreadPool {
public:
  ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i) {
      workers.emplace_back(
        [this] {
          for (;;) {
            std::function<void()> task;
            {
              std::unique_lock<std::mutex> lock(this->queue_mutex);
              this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
              if (this->stop && this->tasks.empty()) return;
              task = std::move(this->tasks.front());
              this->tasks.pop();
            }
            task();
          }
        }
      );
    }
  }

  template<class F, class... Args>
  auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
      tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers) {
      worker.join();
    }
  }

private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

} // namespace rm_auto_aim

#endif // ARMOR_DETECTOR_THREAD_POOL_HPP_