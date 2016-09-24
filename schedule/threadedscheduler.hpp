#pragma once

#ifndef FRAC_NO_THREADS
#include "scheduler.h"
#include "sequentialscheduler.hpp"
#include <thread>
#include <mutex>
#include <deque>
#include <iterator>
#include <condition_variable>

namespace Frac {

class SafeFlag {
public:
	void waitFor(bool value) {
		std::unique_lock<std::mutex> lock(this->_mutex);
		if (this->_flag != value)
			this->_waitCondition.wait(lock);
		this->_flag = !value;
	}
	void set(bool value) {
		this->_mutex.lock();
		this->_flag = value;
		this->_mutex.unlock();
		this->_waitCondition.notify_one();
	}
private:
	std::mutex _mutex;
	std::condition_variable _waitCondition;
	bool _flag = false;
};

template <typename Result>
class _ThreadSchedulerHelper : public AbstractScheduler<Result> {
	using TaskPtr = std::unique_ptr<AbstractTask<Result>>;
public:
	~_ThreadSchedulerHelper() {
		if (_jobsDone > 0)
			std::cout << "jobs done: " << _jobsDone << '\n';
	}
	void addTask(AbstractTask<Result>* task) {
		std::lock_guard<std::mutex> lock(_pendingMutex);
		_pending.emplace_back(TaskPtr(task));
		_waitForTaskFlag.notify_one();
	}
	void waitForAll() {
		while (1) {
			{
				std::lock_guard<std::mutex> lock(_runMutex);
				if (_done) {
					std::unique_lock<std::mutex> lock(_pendingMutex);
					if (!_processing)
						break;
				}
			}
			{
				std::unique_lock<std::mutex> lock(_pendingMutex);
				if (_pending.empty() && !_processing)
					break;
				else
					_taskDoneFlag.wait(lock);
			}
		}
	}
	void mapResults(std::function<void(Result&)> fn) {
		std::lock_guard<std::mutex> lock(_resultMutex);
		for (auto& r : _results)
			fn(r);
	}
	std::vector<Result> allResults() const {
		std::lock_guard<std::mutex> lock(_resultMutex);
		return _results;
	}
	size_t workload() const {
		std::lock_guard<std::mutex> lock(_pendingMutex);
		return _pending.size() + _processing;
	}
	void finish() {
		std::lock_guard<std::mutex> lock(_runMutex);
		_done = true;
	}
	void threadFunc() {
		while (1) {
			{
				std::lock_guard<std::mutex> lock(_runMutex);
				if (_done) {
					std::unique_lock<std::mutex> lock(_pendingMutex);
					if (!_processing)
						break;
				}
			}
			std::unique_ptr<AbstractTask<Result>> task;
			{
				std::unique_lock<std::mutex> lock(_pendingMutex);
				if (_pending.empty() == false) {
					_processing = true;
					task.swap(_pending.back());
					_pending.pop_back();
				} else {
					_processing = false;
					_waitForTaskFlag.wait(lock);
				}
			}
			if (task) {
				task->run();
				{
					std::lock_guard<std::mutex> lock(_resultMutex);
					_results.push_back(task->result());
					++_jobsDone;
				}
				std::unique_lock<std::mutex> lock(_pendingMutex);
				_processing = false;
				_taskDoneFlag.notify_one();
			}
		}
	}
private:
	std::deque<TaskPtr> _pending;
	std::vector<Result> _results;
	mutable std::mutex _pendingMutex;
	mutable std::mutex _resultMutex;
	std::condition_variable _waitForTaskFlag;
	std::condition_variable _taskDoneFlag;
	std::mutex _runMutex;
	bool _done = false;
	size_t _jobsDone = 0;
	bool _processing = false;
};

template <typename Result>
class ThreadedScheduler : public AbstractScheduler<Result> {
public:
	explicit ThreadedScheduler(const size_t numThreads) {
		for (size_t i = 0; i < numThreads; ++i) {
			std::shared_ptr<_ThreadSchedulerHelper<Result>> scheduler(new _ThreadSchedulerHelper<Result>());
			_threadQueue.push_back(scheduler);
		}
	}
	~ThreadedScheduler() {
		for (auto& queue : _threadQueue)
			queue->finish();
		for (auto& thread : _threads)
			thread->detach();
	}
	void addTask(AbstractTask<Result>* task) {
		if (!_threadsStarted) {
			_threadsStarted = true;
			for (size_t i = 0; i < _threadQueue.size(); ++i) {
				_threads.push_back(std::unique_ptr<std::thread>(new std::thread(
					[&]() { _threadQueue[i]->threadFunc(); }
				)));
			}
			std::cout << "starting " << _threadQueue.size() << " threads...\n";
		}
		size_t lowestWorkload = 99999;
		size_t index = 0;
		for (size_t i = 0; i < _threadQueue.size(); ++i) {
			auto threadWorkload = _threadQueue[i]->workload();
			if (threadWorkload < lowestWorkload) {
				index = i;
				lowestWorkload = threadWorkload;
			}
		}
		_threadQueue[index]->addTask(task);
	}
	void waitForAll() {
		for (auto & t : _threadQueue) {
			t->waitForAll();
			t->finish();
		}
	}
	void mapResults(std::function<void(Result&)> fn) {
		for (auto & t : _threadQueue)
			t->mapResults(fn);
	}
	std::vector<Result> allResults() const {
		std::vector<Result> result;
		for (auto& t : _threadQueue) {
			const auto threadResults = t->allResults();
			std::copy(threadResults.begin(), threadResults.end(), std::back_inserter(result));
		}
		return result;
	}
	size_t workload() const {
		return 0;
	}
private:
	std::vector<std::shared_ptr<_ThreadSchedulerHelper<Result>>> _threadQueue;
	std::vector<std::unique_ptr<std::thread>> _threads;
	bool _threadsStarted = false;
};
}

#endif
