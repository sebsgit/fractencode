#ifndef PARTITION_H
#define PARTITION_H

#include "image/image.h"
#include "image/transform.h"
#include "image/metrics.h"
#include "encode/datatypes.h"
#include <vector>

namespace Frac {
    class TransformMatcher;
    class ImageClassifier;

    class PartitionItem {
    public:
        virtual ~PartitionItem() {}
        virtual Image& image() noexcept = 0;
        virtual Image image() const noexcept = 0;
        virtual double distance(const PartitionItem& other, const Metric& m, const Transform& t) const = 0;
        virtual const Point2du pos() const = 0;
        uint32_t width() const noexcept {
            return image().width();
        }
        uint32_t height() const noexcept {
            return image().height();
        }
        Size32u size() const noexcept {
            return image().size();
        }
    };
    using PartitionItemPtr = std::shared_ptr<PartitionItem>;
    class Partition;
    using PartitionPtr = std::shared_ptr<Partition>;

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

    class Partition {
    public:
        Partition() {

        }
        virtual ~Partition() {

        }
        const std::vector<PartitionItemPtr>::const_iterator begin() const {
            return _data.begin();
        }
        const std::vector<PartitionItemPtr>::const_iterator end() const {
            return _data.end();
        }
        std::vector<PartitionItemPtr>::iterator begin() {
            return _data.begin();
        }
        std::vector<PartitionItemPtr>::iterator end() {
            return _data.end();
        }
        size_t size() const {
            return _data.size();
        }
        void push_back(const PartitionItemPtr& p) {
            this->_data.push_back(p);
        }
        void merge(const PartitionPtr& other) {
            _data.insert(end(), other->begin(), other->end());
        }

        virtual grid_encode_data_t estimateMapping(const PartitionPtr& source, const ImageClassifier&, const TransformMatcher&, uint64_t &rejectedMappings) = 0;
    protected:
        virtual item_match_t matchItem(const PartitionItemPtr& p, const PartitionPtr& source, const ImageClassifier&, const TransformMatcher&, uint64_t& rejectedMappings) const;
    protected:
        std::vector<PartitionItemPtr> _data;
    };

    class PartitionCreator {
    public:
        virtual ~PartitionCreator() {}
        virtual PartitionPtr create(const Image&) const = 0;
    };

}

#endif // PARTITION_H
