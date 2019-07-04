#pragma once

#include "encode/classifier.h"
#include "encode/transformmatcher.h"
#include <memory>
#include <atomic>

namespace Frac {

	class TransformEstimator {
	public:
		TransformEstimator(std::shared_ptr<ImageClassifier> classifier, std::shared_ptr<TransformMatcher> matcher, const PartitionPtr& sourcePartition)
			: _classifier(classifier)
			, _matcher(matcher)
			, _source(sourcePartition)
		{
			this->_rejectedMappings = 0;
		}
		item_match_t estimate(const PartitionItemPtr& targetItem) const {
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
			return result;
		}
		int rejectedMappings() const noexcept {
			return this->_rejectedMappings;
		}
	private:
		std::shared_ptr<ImageClassifier> _classifier;
		std::shared_ptr<TransformMatcher> _matcher;
		PartitionPtr _source;
		PartitionPtr _target;
		mutable std::atomic_int _rejectedMappings;
	};
}
