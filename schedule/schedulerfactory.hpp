#pragma once
#include "sequentialscheduler.hpp"
#include "threadedscheduler.hpp"

namespace Frac {
template <typename Result>
class SchedulerFactory {
public:
	static AbstractScheduler<Result>* create() {
		return new ThreadedScheduler<Result>(std::thread::hardware_concurrency() - 1);
		//return new SequentialScheduler<Result>();
	}
};
}
