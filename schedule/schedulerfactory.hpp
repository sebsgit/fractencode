#pragma once
#include "sequentialscheduler.hpp"
#include "threadedscheduler.hpp"

namespace Frac {
template <typename Result>
class SchedulerFactory {
public:
	static AbstractScheduler<Result>* create() {
#ifdef FRAC_NO_THREADS
		return new SequentialScheduler<Result>();
#else
		return new ThreadedScheduler<Result>(std::thread::hardware_concurrency() - 1);
#endif
	}
};
}
