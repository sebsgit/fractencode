#include "gridpartition.h"
#include "encode/classifier.h"
#include "encode/transformmatcher.h"
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

grid_encode_data_t GridPartition::estimateMapping(const PartitionPtr &source, const ImageClassifier& classifier, const TransformMatcher& matcher, uint64_t& rejectedMappings) {
    grid_encode_data_t result;
    for (auto it : this->_data) {
        item_match_t match = this->matchItem(it, source, classifier, matcher, rejectedMappings);
        encode_item_t enc;
        enc.x = it->pos().x();
        enc.y = it->pos().y();
        enc.w = it->image().width();
        enc.h = it->image().height();
        enc.match = match;
        result.encoded.push_back(enc);
        std::cout << it->pos().x() << ", " << it->pos().y() << " --> " << match.x << ',' << match.y << " d: " << match.score.distance << "\n";
        std::cout << "s, o: " << match.score.contrast << ' ' << match.score.brightness << "\n";
    }
    return result;
}

item_match_t Partition::matchItem(const PartitionItemPtr& a, const PartitionPtr &source, const ImageClassifier& classifier, const TransformMatcher& matcher, uint64_t &rejectedMappings) const {
    item_match_t result;
    uint32_t i = 0;
    for (auto it : source->_data) {
        if (classifier.compare(a->image(), it->image())) {
            auto score = matcher.match(a, it);
            if (score.distance < result.score.distance) {
                result.score = score;
                result.x = it->pos().x();
                result.y = it->pos().y();
            }
            if (matcher.checkDistance(result.score.distance))
                break;
        } else {
            rejectedMappings++;
        }
        ++i;
    }
    return result;
}
