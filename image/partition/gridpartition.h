#ifndef GRIDPARTITION_H
#define GRIDPARTITION_H

#include "partition.h"

namespace Frac {

class GridItem : public PartitionItem {
public:
    GridItem(const Image& source, const uint32_t x, const uint32_t y, const Size32u& s)
        :_pos(x, y)
        ,_image(source.slice(x, y, s.x(), s.y()))
    {
    }
    GridItem(const Image& source, const uint32_t x, const uint32_t y)
        :_pos(x, y)
        ,_image(source)
    {
    }
    ~GridItem() {}
    double distance(const PartitionItem& other, const Metric& m, const Transform& t) const override {
        return m.distance(other.image(),this->image(),t);
    }
    Image& image() noexcept override {
        return _image;
    }
    Image image() const noexcept override {
        return _image;
    }
    const Point2du pos() const noexcept {
        return _pos;
    }
private:
    const Point2du _pos;
    Image _image;
};

class GridPartition : public Partition {
public:
    grid_encode_data_t estimateMapping(const PartitionPtr& source, const ImageClassifier&, const TransformMatcher&, uint64_t &rejectedMappings) override;
};

class GridPartitionCreator : public PartitionCreator {
public:
    GridPartitionCreator(const Size32u& itemSize, const Size32u& offset)
        : _size(itemSize)
        , _offset(offset)
    {

    }
    ~GridPartitionCreator() {

    }
    PartitionPtr create(const Image& image) const override;
protected:
    const Size32u _size;
    const Size32u _offset;
};

class AdaptativeGridPartitionCreator : public GridPartitionCreator {
public:
    AdaptativeGridPartitionCreator(const Size32u& baseSize, const Size32u& offset)
        :GridPartitionCreator(baseSize, offset)
    {

    }
    PartitionPtr create(const Image& image) const override;
};

}

#endif // GRIDPARTITION_H
