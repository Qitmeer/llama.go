#pragma once
#include <queue>
#include <mutex>

class SafeQueue {
public:
    void enqueue(std::string task);

    bool dequeue(std::string& task);

    void stop();

private:
    std::queue<std::string> m_queue;
    std::mutex m_mtx;
    std::condition_variable m_cv;
    bool m_stop = false;
};