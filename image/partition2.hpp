#pragma once

#include "utils/point2d.hpp"
#include "utils/size.hpp"
#include "utils/Assert.hpp"

#include <vector>
#include <functional>

namespace Frac2 {
    using namespace Frac;
    
    class GridItemBase {
    public:
        Point2du origin;
        Size32u size;
    };

    template <typename ExtraDataIn = std::nullptr_t>
    class GridItem : public GridItemBase {
    public:
        using ExtraData = ExtraDataIn;
        ExtraData data;
    };

    template <typename Item>
    class GridPartition {
    public:
        const auto& items() const noexcept { return this->_items; }
        void reserve(size_t n) { this->_items.reserve(n); }
        void add(const Point2du& origin, const Size32u& size, typename Item::ExtraData&& d)
        {
            this->_items.push_back(Item{origin, size, std::move(d)});
        }
        void add(const Point2du& origin, const Size32u& size)
        {
            this->_items.push_back(Item{ origin, size });
        }
    private:
        std::vector<Item> _items;
    };

    struct GridItemData {
        // any grid per-item data lands here
    };

    using UniformGridItem = GridItem<GridItemData>;
    using UniformGrid = GridPartition<UniformGridItem>;

    /**
        Creates a uniform grid partition over the specified size.
        @param imageSize Total size of the image.
        @param itemSize Size of each item in this grid.
        @param itemOffset Offset between each grid element.
    */
    template <typename Item = UniformGridItem>
    GridPartition<Item> createUniformGrid(
        const Size32u& imageSize,
        const Size32u& itemSize, 
        const Size32u& itemOffset,
        const std::function<typename Item::ExtraData(const Point2du&, const Size32u&)>& callback = [](const Point2du& /*itemPos*/, const Size32u& /*itemSize*/) -> typename Item::ExtraData {
            return typename Item::ExtraData{};
        }
    )
    {
        FRAC_ASSERT(imageSize.isAligned(itemSize.x(), itemSize.y()) && "can't create grid partition on unaligned image!");
        FRAC_ASSERT(imageSize.isAligned(itemOffset.x(), itemOffset.y()) && "can't create grid partition with unaligned offset!");

        GridPartition<Item> result;
        uint32_t x = 0, y = 0;
        do {
            result.add({ x, y }, itemSize, callback({x, y}, itemSize));
            x += itemOffset.x();
            if (x + itemSize.x() > imageSize.x()) {
                x = 0;
                y += itemOffset.y();
                if (y + itemSize.y() > imageSize.y())
                    break;
            }
        } while (1);
        return result;
    }
}
