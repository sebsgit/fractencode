#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <vector>
#include <functional>

namespace Frac {

template <typename Result>
class AbstractTask {
public:
	virtual ~AbstractTask() {}
	virtual void run() = 0;
	virtual Result result() const = 0;
};

template <typename Result>
class LambdaTask : public AbstractTask<Result> {
public:
	explicit LambdaTask(std::function<Result(void)> fn) 
		:_task(fn)
	{}
	void run() override {
		this->_result = this->_task();
	}
	Result result() const override {
		return this->_result;
	}
private:
	std::function<Result(void)> _task;
	Result _result;
};

template<typename Result>
class AbstractScheduler {
public:
	virtual ~AbstractScheduler() { }
	virtual void addTask(AbstractTask<Result>* task) = 0;
	virtual void waitForAll() = 0;
	virtual void mapResults(std::function<void(Result&)> fn) = 0;
	virtual std::vector<Result> allResults() const = 0;
	virtual size_t workload() const = 0;
};

}

#endif

