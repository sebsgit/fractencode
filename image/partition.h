#ifndef PARTITION_H
#define PARTITION_H

#include "image/image.h"
#include "image/transform.h"
#include "image/metrics.h"
#include <vector>

namespace Frac {
    class PartitionItem {
    public:
        virtual ~PartitionItem() {}
        virtual Image& image() noexcept = 0;
        virtual Image image() const noexcept = 0;
        virtual double distance(const PartitionItem& other, const Metric& m, const Transform& t) const = 0;
        virtual const Point2du pos() const = 0;
    };
    using PartitionItemPtr = std::shared_ptr<PartitionItem>;
    using PartitionData = std::vector<PartitionItemPtr>;

    class Partition {
    public:
        virtual ~Partition() {}
        virtual PartitionData create(const Image&) = 0;
    };


    class GridItem : public PartitionItem {
    public:
        GridItem(const Image& source, const uint32_t x, const uint32_t y, const Size32u& s)
            :_pos(x, y)
            ,_image(source.slice(x, y, s.x(), s.y()))
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
        GridPartition(const Size32u& itemSize, const Size32u& offset)
            : _size(itemSize)
            , _offset(offset)
        {

        }
        ~GridPartition() {

        }
        PartitionData create(const Image& image) override {
            assert(image.size().isAligned(_size.x(), _size.y()) && "cant create grid partition on unaligned image !");
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
    private:
        const Size32u _size;
        const Size32u _offset;
    };
}

#endif // PARTITION_H
