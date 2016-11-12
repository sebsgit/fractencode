#include "gridpartition.h"
#include "encode/classifier.h"
#include <iostream>

using namespace Frac;

PartitionPtr GridPartitionCreator::create(const Image& image) const {
	assert(image.size().isAligned(_size.x(), _size.y()) && "can't create grid partition on unaligned image!");
	assert(image.size().isAligned(_offset.x(), _offset.y()) && "can't create grid partition with unaligned offset!");
	PartitionPtr result(new GridPartition);
	uint32_t x = 0, y = 0;
	do {
		result->push_back(PartitionItemPtr(new GridItem(image, x, y, _size)));
		x += _offset.x();
		if (x + _size.x() > image.width()) {
			x = 0;
			y += _offset.y();
			if (y + _size.y() > image.height())
				break;
		}
	} while (1);
	return result;
}

PartitionPtr AdaptativeGridPartitionCreator::create(const Image& image) const {
	assert(image.size().isAligned(_size.x(), _size.y()) && "can't create grid partition on unaligned image!");
	assert(image.size().isAligned(_offset.x(), _offset.y()) && "can't create grid partition with unaligned offset!");
	PartitionPtr result(new GridPartition);
	PartitionPtr textureRegions(new GridPartition);
	uint32_t x = 0, y = 0;
	TextureClassifier classifier;
	const auto size2 = _size / 2;
	do {
		auto patch = image.slice(x, y, _size.x(), _size.y());
		if (classifier.isFlat(patch)) {
			result->push_back(PartitionItemPtr(new GridItem(patch, x, y)));
		} else {
			textureRegions->push_back(PartitionItemPtr(new GridItem(image, x, y, size2)));
			textureRegions->push_back(PartitionItemPtr(new GridItem(image, x + size2.x(), y, size2)));
			textureRegions->push_back(PartitionItemPtr(new GridItem(image, x, y + size2.y(), size2)));
			textureRegions->push_back(PartitionItemPtr(new GridItem(image, x + size2.x(), y + size2.y(), size2)));
			x += _offset.x();
		}
		x += _offset.x();
		if (x + _size.x() > image.width()) {
			x = 0;
			y += _offset.y();
			if (y + _size.y() > image.height())
				break;
		}
	} while (1);
	result->merge(textureRegions);
	return result;
}
