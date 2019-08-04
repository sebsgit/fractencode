#pragma once

#include "encode/datatypes.h"
#include "encode/encode_parameters.h"
#include "encode/TransformEstimator2.hpp"

#include <iostream>
#include <string>
#include <sstream>
#include <thread>

namespace Frac2 {
    class ProgressReporter2 {
    public:
        virtual ~ProgressReporter2() {}
        virtual void log(size_t done, size_t total) = 0;
    };

    class StdoutReporter2 : public ProgressReporter2 {
    public:
        void log(size_t done, size_t total) override
        {
            auto now = std::chrono::system_clock::now();
            std::chrono::duration<double> seconds = now - _lastPrintout;
            if (seconds.count() > 0.3) {
                _lastPrintout = now;
                const double percent = (100.0 * done) / total;
                std::stringstream ss;
                ss << percent;
                const auto toPrint = ss.str();
                rewind();
                this->_lastPrintLength = toPrint.size();
                std::cout << toPrint;
            }
        }
    private:
        void rewind()
        {
            while (_lastPrintLength > 0) {
                --_lastPrintLength;
                std::cout << '\b';
            }
            std::fflush(stdout);
        }
    private:
        size_t _lastPrintLength = 0;
        std::chrono::system_clock::time_point _lastPrintout;
    };

    class AbstractEncodingEngine2 {
    public:
        AbstractEncodingEngine2(
            const encode_parameters_t& params,
            const ImagePlane& sourceImage,
            const UniformGrid& sourceGrid)
            : _parameters(params)
            , _image(sourceImage)
            , _source(sourceGrid)
        {}
        virtual ~AbstractEncodingEngine2() {
            std::cout << this->_name << ", tasks done: " << this->_tasksDone << '\n';
        }
        virtual void encode(const UniformGridItem& targetItem) {
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
        virtual encode_item_t encode_impl(const UniformGridItem& targetItem) const = 0;
    protected:
        const encode_parameters_t _parameters;
        const ImagePlane& _image;
        const UniformGrid& _source;
    private:
        std::vector<encode_item_t> _result;
        std::string _name;
        int _tasksDone = 0;
    };


    class CpuEncodingEngine2 : public AbstractEncodingEngine2 {
    public:
        CpuEncodingEngine2(const encode_parameters_t& params,
            const ImagePlane& sourceImage,
            const UniformGrid& sourceGrid,
            const TransformEstimator2& estimator)
            : AbstractEncodingEngine2(params, sourceImage, sourceGrid)
            , _estimator(estimator)
        {

        }
    protected:
        encode_item_t encode_impl(const UniformGridItem& targetItem) const override {
            auto itemMatch = this->_estimator.estimate(targetItem);
            encode_item_t enc;
            enc.x = targetItem.origin.x();
            enc.y = targetItem.origin.y();
            enc.w = targetItem.size.x();
            enc.h = targetItem.size.y();
            enc.match = itemMatch;
            return enc;
        }
    private:
        const TransformEstimator2& _estimator;
    };


    class EncodingEngineCore2 {
    public:
        EncodingEngineCore2(const encode_parameters_t& params, const ImagePlane& image, const UniformGrid& gridSource, const TransformEstimator2& estimator, ProgressReporter2* reporter);
        void encode(const UniformGrid& gridTarget) {
            size_t jobQueueIndex = 0;
            std::vector<std::unique_ptr<std::thread>> threads;
            std::mutex queueMutex;
            std::mutex doneMutex;
            std::condition_variable queueEmpty;
            int tasksDone = 0;
            const auto & jobQueue = gridTarget.items();
            for (size_t i = 0; i < this->_engines.size(); ++i) {
                auto fn = [&, i]() {
                    this->_engines[i]->init();
                    while (1) {
                        UniformGridItem task;
                        bool hasTask = false;
                        {
                            std::lock_guard<std::mutex> lock(queueMutex);
                            if (jobQueueIndex < jobQueue.size()) {
                                hasTask = true;
                                task = jobQueue[jobQueueIndex];
                                ++jobQueueIndex;
                                _reporter->log(jobQueueIndex, jobQueue.size());
                            }
                        }
                        if (hasTask) {
                            this->_engines[i]->encode(task);
                        }
                        else {
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
        std::vector<std::unique_ptr<AbstractEncodingEngine2>> _engines;
        const TransformEstimator2& _estimator;
        grid_encode_data_t _result;
        ProgressReporter2* _reporter; // not owned
    };
} // namespace Frac2
