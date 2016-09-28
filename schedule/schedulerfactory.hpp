#pragma once
#include "sequentialscheduler.hpp"
#include "threadedscheduler.hpp"
#include <memory>

namespace Frac {
template <typename Result>
class SchedulerFactory {
public:
	static std::unique_ptr<AbstractScheduler<Result>> create() {
		//return std::make_unique<SequentialScheduler<Result>>();
#ifdef FRAC_NO_THREADS
		return std::make_unique<SequentialScheduler<Result>>();
#else
		return std::make_unique<ThreadedScheduler<Result>>(std::thread::hardware_concurrency() - 1);
#endif
	}
};
}
