#pragma once

#include "encode/classifier.h"
#include "schedule/schedulerfactory.hpp"
#include "encode/transformmatcher.h"
#include <memory>
#include <atomic>

namespace Frac {

	class TransformEstimator {
	public:
		TransformEstimator(std::shared_ptr<ImageClassifier> classifier, std::shared_ptr<TransformMatcher> matcher, const PartitionPtr& sourcePartition, const PartitionPtr& targetPartition)
			: _classifier(classifier)
			, _matcher(matcher)
			, _source(sourcePartition)
			, _target(targetPartition)
			, _scheduler(SchedulerFactory<encode_item_t>::create())
		{}
		void estimate(const Image& image) {
			for (auto targetItem : *this->_target) {
				auto fn = [this, targetItem]() {
					item_match_t result;
					for (auto src : *this->_source) {
						if (this->_classifier->compare(src, targetItem)) {
							auto score = this->_matcher->match(targetItem, src);
							if (score.distance < result.score.distance) {
								result.score = score;
								result.x = src->pos().x();
								result.y = src->pos().y();
								result.sourceItemSize = src->sourceSize();
							}
							if (this->_matcher->checkDistance(result.score.distance))
								break;
						}
						else {
							++this->_rejectedMappings;
						}
					}
					encode_item_t enc;
					enc.x = targetItem->pos().x();
					enc.y = targetItem->pos().y();
					enc.w = targetItem->image().width();
					enc.h = targetItem->image().height();
					enc.match = result;
					return enc;
				};
				this->_scheduler->addTask(new LambdaTask<encode_item_t>(fn));
			}
		}
		grid_encode_data_t result() const {
			this->_scheduler->waitForAll();
			return grid_encode_data_t{ this->_scheduler->allResults() };
		}
		int rejectedMappings() const noexcept {
			return this->_rejectedMappings;
		}
	private:
		std::shared_ptr<ImageClassifier> _classifier;
		std::shared_ptr<TransformMatcher> _matcher;
		PartitionPtr _source;
		PartitionPtr _target;
		std::unique_ptr<AbstractScheduler<encode_item_t>> _scheduler;
		std::atomic_int _rejectedMappings = 0;
	};
}
