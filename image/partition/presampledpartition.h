#pragma once
#include "partition.h"
#include "gridpartition.h"

namespace Frac {

	//TODO support pre-sampling to lower resolutions (x4, x8)
class PreSampledPartitionCreator : public GridPartitionCreator {
public:
	PreSampledPartitionCreator(const Size32u& size, const Size32u& offset) : GridPartitionCreator(size, offset) {}
	PartitionPtr create(const Image& image) const override {
		assert(image.size().isAligned(_size.x(), _size.y()) && "can't create pre-sampled partition on unaligned image!");
		assert(image.size().isAligned(_offset.x(), _offset.y()) && "can't create grid partition with unaligned offset!");
		Image presampled = Transform().resize(image, image.size() / 2, Transform::Bilinear);
		PartitionPtr result(new GridPartition());
		for (uint32_t y = 0; y < image.height(); y += _offset.y()) {
			for (uint32_t x = 0; x < image.width(); x += _offset.x()) {
				if (x + _size.x() > image.width() || y + _size.y() > image.height())
					continue;
				auto sampled = presampled.slice(x / 2, y / 2, _size.x() / 2, _size.y() / 2, Image::NoCache);
				result->push_back(PartitionItemPtr(new GridItem(image, x, y, _size, std::move(sampled))));
			}
		}
		return result;
	}
};

}
