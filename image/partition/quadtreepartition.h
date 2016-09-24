#ifndef QUADTREEPARTITION_H
#define QUADTREEPARTITION_H

#include "partition.h"

namespace Frac {
class QuadtreePartitionCreator: public PartitionCreator {
public:
    QuadtreePartitionCreator(const Size32u& baseSize, const Size32u& minSize)
        :_baseSize(baseSize)
        ,_minSize(minSize)
    {}
    PartitionPtr create(const Image& image) const override;
private:
    const Size32u _baseSize;
    const Size32u _minSize;
};

class QuadtreePartition : public Partition {
public:
    explicit QuadtreePartition(const Size32u& minSize) : _minSize(minSize) {

    }
    grid_encode_data_t estimateMapping(const PartitionPtr& source, const ImageClassifier&, const TransformMatcher&, uint64_t &rejectedMappings) override;
private:
    bool canSplit(const PartitionItemPtr& it) const noexcept;
    void estimateMappingImpl(const std::vector<PartitionItemPtr>& data, const PartitionPtr& source, const ImageClassifier&, const TransformMatcher&, uint64_t &rejectedMappings, grid_encode_data_t& result);
private:
    const Size32u _minSize;
};

}

#endif // QUADTREEPARTITION_H
