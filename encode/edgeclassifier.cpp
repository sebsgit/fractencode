#include "encode/edgeclassifier.h"
#include "process/gaussian5x5.h"
#include "process/sobel.h"
#include <cassert>
#include <iostream>

using namespace Frac;

class EdgeClassifier::Data {
public:
	explicit Data(Image image) {
		assert(image.size().isAligned(_patchSize, _patchSize) && "can't process edge info on unaligned image!");
		_stride = (image.width() / _patchSize);
		const uint32_t nPatches = _stride * (image.height() / _patchSize);
		_data = Buffer<uint32_t>::alloc(nPatches);
		_data->memset(0);
		image = GaussianBlur5x5().process(image);
		auto sobelData = SobelOperator().calculate(image);
		NonMaximumSuppressionOperator([&](uint32_t x, uint32_t y, bool isStrongEdge) {
			(void)isStrongEdge;
			x /= _patchSize;
			y /= _patchSize;
			_data->get()[x + y * _stride] = 1;
		}).process(sobelData, image.width(), image.height());
	}
	uint32_t score(const Point2du& p) const {
		return score(p.x(), p.y());
	}
	uint32_t score(uint32_t imageX, uint32_t imageY) const {
		return _data->get()[(imageX / _patchSize) + _stride * (imageY / _patchSize)];
	}
private:
	const uint32_t _patchSize = 16;
	uint32_t _stride = 0;
	AbstractBufferPtr<uint32_t> _data;
};

EdgeClassifier::EdgeClassifier(Image image)
	:_data(new Data(image))
{

}

EdgeClassifier::~EdgeClassifier() {

}

bool EdgeClassifier::compare(const PartitionItemPtr& targetItem, const PartitionItemPtr& sourceItem) const {
	return _data->score(targetItem->pos()) == _data->score(sourceItem->pos());
}

bool EdgeClassifier::compare(const Image&, const Image&) const {
	std::cout << "images should not be compared directly with edge classifier!\n";
	return false;
}
