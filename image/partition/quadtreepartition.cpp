#include "quadtreepartition.h"
#include "gridpartition.h"
#include "encode/transformmatcher.h"
#include <iostream>

using namespace Frac;

PartitionPtr QuadtreePartitionCreator::create(const Image& image) const {
	assert(this->_baseSize.x() > this->_minSize.x());
	assert(this->_baseSize.y() > this->_minSize.y());
	GridPartitionCreator gridHelper(this->_baseSize, this->_baseSize);
	PartitionPtr grid = gridHelper.create(image);
	PartitionPtr quadtree(new QuadtreePartition(this->_minSize));
	quadtree->merge(grid);
	return quadtree;
}

grid_encode_data_t QuadtreePartition::estimateMapping(const PartitionPtr& source, const ImageClassifier& classifier, const TransformMatcher& matcher, uint64_t &rejectedMappings) {
	grid_encode_data_t result;
	this->estimateMappingImpl(this->_data, source, classifier, matcher, rejectedMappings, result);
	return result;
}

void QuadtreePartition::estimateMappingImpl(const std::vector<PartitionItemPtr>& data, const PartitionPtr& source, const ImageClassifier& classifier, const TransformMatcher& matcher, uint64_t &rejectedMappings, grid_encode_data_t& result) {
	std::vector<PartitionItemPtr> nextLevel;
	for (auto it : data) {
		item_match_t match = this->matchItem(it, source, classifier, matcher, rejectedMappings);
		if (matcher.checkDistance(match.score.distance) || !this->canSplit(it)) {
			encode_item_t enc;
			enc.x = it->pos().x();
			enc.y = it->pos().y();
			enc.w = it->image().width();
			enc.h = it->image().height();
			enc.match = match;
			result.encoded.push_back(enc);
			std::cout << it->pos().x() << ", " << it->pos().y() << " --> " << match.x << ',' << match.y << " d: " << match.score.distance << "\n";
			std::cout << "s, o: " << match.score.contrast << ' ' << match.score.brightness << "\n";
		} else {
			const auto size = it->size() / 2;
			const auto x = it->pos().x();
			const auto y = it->pos().y();
			const auto w = size.x();
			const auto h = size.y();
			const Image topLeft = it->image().slice(0, 0, w, h);
			const Image topRight = it->image().slice(w, 0, w, h);
			const Image bottomLeft = it->image().slice(0, h, w, h);
			const Image bottomRight = it->image().slice(w, h, w, h);
			PartitionItemPtr tl(new GridItem(topLeft, x, y));
			PartitionItemPtr tr(new GridItem(topRight, x + w, y + 0));
			PartitionItemPtr bl(new GridItem(bottomLeft, x + 0, y + size.y()));
			PartitionItemPtr br(new GridItem(bottomRight, x + size.x(), y + size.y()));
			nextLevel.push_back(tl);
			nextLevel.push_back(tr);
			nextLevel.push_back(bl);
			nextLevel.push_back(br);
		}
	}
	if (nextLevel.empty() == false) {
		this->estimateMappingImpl(nextLevel, source, classifier, matcher, rejectedMappings, result);
		this->_data.insert(_data.end(), nextLevel.begin(), nextLevel.end());
	}
}

bool QuadtreePartition::canSplit(const PartitionItemPtr &it) const noexcept {
	return it->width() > this->_minSize.x() && it->height() > this->_minSize.y();
}
