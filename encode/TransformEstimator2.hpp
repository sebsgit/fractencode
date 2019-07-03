#pragma once

#include "image/Image2.hpp"
#include "image/partition2.hpp"
#include "encode/transformmatcher.h"
#include "encode/Classifier2.hpp"


namespace Frac2 {
    class TransformEstimator2 {
    public:
        TransformEstimator2(
            const ImagePlane& sourceImage,
            const ImagePlane& targetImage,
            std::unique_ptr<Classifier2>&& classifier,
            const std::shared_ptr<TransformMatcher>& matcher,
            const UniformGrid& sourceGrid)
            : _sourceImage(sourceImage)
            , _targetImage(targetImage)
            , _sourceGrid(sourceGrid)
            , _classifier(std::move(classifier))
            , _matcher(matcher)
        {
			this->_rejectedMappings = 0;
		}
		
        item_match_t estimate(const Frac2::UniformGridItem& targetItem) const {
            item_match_t result;
            for (const auto& sourcePatch : this->_sourceGrid.items()) {
                if (this->_classifier->compare(sourcePatch, targetItem)) {
                    auto score = this->_matcher->match(this->_sourceImage, sourcePatch, this->_targetImage, targetItem);
                    if (score.distance < result.score.distance) {
                        result.score = score;
                        result.x = sourcePatch.origin.x();
                        result.y = sourcePatch.origin.y();
                        result.sourceItemSize = sourcePatch.size;
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
        const ImagePlane& _sourceImage;
        const ImagePlane& _targetImage;
        std::unique_ptr<Classifier2> _classifier;
        std::shared_ptr<TransformMatcher> _matcher;
        const UniformGrid& _sourceGrid;
        mutable std::atomic_int _rejectedMappings;
    };
}