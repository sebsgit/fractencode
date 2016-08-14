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

    class PartitionData {
    public:
        PartitionData() {

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
        const PartitionItemPtr& at(const size_t i) const {
            return _data.at(i);
        }

    private:
        std::vector<PartitionItemPtr> _data;
    };

    class PartitionCreator {
    public:
        virtual ~PartitionCreator() {}
        virtual PartitionData create(const Image&) = 0;
    };

}

#endif // PARTITION_H
