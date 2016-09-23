#pragma once

#include "scheduler.h"
#include <memory>

namespace Frac {
template<typename Result>
class SequentialScheduler : public AbstractScheduler<Result> {
public:
	using TaskPtr = std::unique_ptr<AbstractTask<Result>>;
	~SequentialScheduler() {
	}
	void addTask(AbstractTask<Result>* task) override {
		_tasks.emplace_back(TaskPtr(task));
	}
	void waitForAll() override {
		for (auto& t : _tasks) {
			t->run();
			_results.emplace_back(t->result());
		}
	}
	void mapResults(std::function<void(Result&)> fn) override {
		for (auto& t : _results)
			fn(t);
	}
	std::vector<Result> allResults() const override {
		return _results;
	}
	size_t workload() const override {
		return _tasks.size();
	}
private:
	std::vector<TaskPtr> _tasks;
	std::vector<Result> _results;
};
}