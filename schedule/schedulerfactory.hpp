#pragma once
#include "sequentialscheduler.hpp"
#include "threadedscheduler.hpp"
#include <memory>

namespace Frac {
template <typename Result>
class SchedulerFactory {
public:
	static std::unique_ptr<AbstractScheduler<Result>> create(size_t maxConcurrentTasks = std::thread::hardware_concurrency() - 1) {
		//return std::make_unique<SequentialScheduler<Result>>();
#ifdef FRAC_NO_THREADS
		return std::make_unique<SequentialScheduler<Result>>();
#else
		return std::make_unique<ThreadedScheduler<Result>>(maxConcurrentTasks);
#endif
	}
};
}
