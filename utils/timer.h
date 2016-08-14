#ifndef TIMER_H
#define TIMER_H

#include <chrono>

namespace Frac {
class Timer {
public:
    Timer() noexcept {}
    void start() {
        _start = std::chrono::high_resolution_clock::now();
    }
    double elapsed() const {
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - _start).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;
};
}

#endif // TIMER_H
