#ifndef GRIDPARTITION_H
#define GRIDPARTITION_H

#include "partition.h"

namespace Frac {

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
