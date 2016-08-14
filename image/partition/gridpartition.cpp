#include "gridpartition.h"

using namespace Frac;

PartitionData GridPartitionCreator::create(const Image& image) {
    assert(image.size().isAligned(_size.x(), _size.y()) && "can't create grid partition on unaligned image!");
    assert(image.size().isAligned(_offset.x(), _offset.y()) && "can't create grid partition with unaligned offset!");
    PartitionData result;
    uint32_t x = 0, y = 0;
    do {
        result.push_back(PartitionItemPtr(new GridItem(image, x, y, _size)));
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
