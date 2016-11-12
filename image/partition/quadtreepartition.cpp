#include "quadtreepartition.h"
#include "gridpartition.h"
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

bool QuadtreePartition::canSplit(const PartitionItemPtr &it) const noexcept {
	return it->width() > this->_minSize.x() && it->height() > this->_minSize.y();
}
