#pragma once

#include "datatypes.h"
#include "partition.h"
#include "TransformEstimator.h"
#include "encode_parameters.h"

#include <thread>
#include <iostream>
#include <iterator>
#include <sstream>

namespace Frac {
class AbstractEncodingEngine {
public:
	AbstractEncodingEngine(const encode_parameters_t& params, const Image& image, const PartitionPtr& sourceGrid)
		: _parameters(params)
		, _image(image)
		, _source(sourceGrid) 
	{}
	virtual ~AbstractEncodingEngine() {
		std::cout << this->_name << ", tasks done: " << this->_tasksDone << '\n';
	}
	virtual void encode(const PartitionItemPtr& targetItem) {
		this->_result.push_back(this->encode_impl(targetItem));
		++this->_tasksDone;
	}
	void setName(const std::string& name) {
		this->_name = name;
	}
	std::vector<encode_item_t> result() const {
		return _result;
	}
	virtual void init() { }
	virtual void finalize() { }
protected:
	virtual encode_item_t encode_impl(const PartitionItemPtr& targetItem) const = 0;
protected:
	const encode_parameters_t _parameters;
	const Image _image;
	const PartitionPtr _source;
private:
	std::vector<encode_item_t> _result;
	std::string _name;
	int _tasksDone = 0;
};

class CpuEncodingEngine : public AbstractEncodingEngine {
public:
	CpuEncodingEngine(const encode_parameters_t& params, const Image& image, const PartitionPtr& gridSource, const std::shared_ptr<TransformEstimator>& estimator)
		: AbstractEncodingEngine(params, image, gridSource)
		, _estimator(estimator)
	{

	}
protected:
	encode_item_t encode_impl(const PartitionItemPtr& targetItem) const override {
		auto itemMatch = this->_estimator->estimate(targetItem);
		encode_item_t enc;
		enc.x = targetItem->pos().x();
		enc.y = targetItem->pos().y();
		enc.w = targetItem->image().width();
		enc.h = targetItem->image().height();
		enc.match = itemMatch;
		return enc;
	}
private:
	const std::shared_ptr<TransformEstimator> _estimator;
};

class EncodingEngineCore {
public:
	EncodingEngineCore(const encode_parameters_t& params, const Image& image, const PartitionPtr& gridSource, const std::shared_ptr<TransformEstimator> estimator);
	void encode(const PartitionPtr& gridTarget) {
		size_t jobQueueIndex = 0;
		std::vector<std::unique_ptr<std::thread>> threads;
		std::mutex queueMutex;
		std::mutex doneMutex;
		std::condition_variable queueEmpty;
		int tasksDone = 0;
		auto jobQueueStart = gridTarget->begin();
		for (size_t i = 0; i < this->_engines.size(); ++i) {
			auto fn = [&, i]() {
				this->_engines[i]->init();
				while (1) {
					PartitionItemPtr task;
					{
						std::lock_guard<std::mutex> lock(queueMutex);
						if (jobQueueIndex < gridTarget->size()) {
							task = *(jobQueueStart + jobQueueIndex);
							++jobQueueIndex;
						}
					}
					if (task) {
						this->_engines[i]->encode(task);
					} else {
						this->_engines[i]->finalize();
						std::unique_lock<std::mutex> lock(doneMutex);
						++tasksDone;
						queueEmpty.notify_one();
						break;
					}
					std::this_thread::yield();
				}
			};
			threads.push_back(std::unique_ptr<std::thread>(new std::thread(fn)));
		}
		while (1) {
			std::unique_lock<std::mutex> lock(doneMutex);
			queueEmpty.wait(lock);
			if (tasksDone == threads.size())
				break;
			std::this_thread::yield();
		}
		for (auto& thread : threads)
			thread->join();
		for (auto& engine : this->_engines) {
			const auto part = engine->result();
			std::copy(part.begin(), part.end(), std::back_inserter(this->_result.encoded));
		}
	//	std::cout << _result.encoded[0].match.score.distance << ' ' << _result.encoded[0].match.score.brightness << ' ' << _result.encoded[0].match.score.contrast << '\n';
	//	std::cout << _result.encoded[0].match.x << ' ' << _result.encoded[0].match.y << ' ' << _result.encoded[0].match.sourceItemSize.x() << ' ' << _result.encoded[0].match.sourceItemSize.y() << '\n';
	}
	const grid_encode_data_t result() const {
		return this->_result;
	}
private:
	std::vector<std::unique_ptr<AbstractEncodingEngine>> _engines;
	const std::shared_ptr<TransformEstimator> _estimator;
	grid_encode_data_t _result;
};
}