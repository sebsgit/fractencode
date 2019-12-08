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

        /// @return item in top left corner with half the original size
        GridItemBase topLeft() const noexcept {
            return GridItemBase{origin, size / 2};
        }
        GridItemBase topRight() const noexcept {
            return GridItemBase{ origin + Point2du{size.x() / 2, 0}, size / 2 };
        }
        GridItemBase bottomLeft() const noexcept {
            return GridItemBase{ origin + Point2du{0, size.y() / 2}, size / 2 };
        }
        GridItemBase bottomRight() const noexcept {
            return GridItemBase{ origin + Point2du{size.x() / 2, size.y() / 2}, size / 2 };
        }
    };

    template <typename ExtraDataIn, bool isEmpty = std::is_empty_v<ExtraDataIn>>
    class GridItem;

    template <typename ExtraDataIn>
    class GridItem<ExtraDataIn, false> : public GridItemBase {
    public:
        using ExtraData = ExtraDataIn;
        ExtraData data;

        GridItem() noexcept : GridItemBase{ Point2du{}, Size32u{} }
        {}
        GridItem(const Point2du& origin, const Size32u& size) noexcept : GridItemBase{ origin, size }
        {
        }
        GridItem(const Point2du& origin, const Size32u& size, ExtraData&& d) noexcept
            :GridItemBase{ origin, size }
            , data(std::move(d))
        {
        }
    };

    template <typename ExtraDataIn>
    class GridItem<ExtraDataIn, true> : public GridItemBase {
    public:
        using ExtraData = ExtraDataIn;

        GridItem() noexcept : GridItemBase{ Point2du{}, Size32u{} }
        {}
        GridItem(const Point2du& origin, const Size32u& size) noexcept : GridItemBase{ origin, size }
        {
        }
        GridItem(const Point2du& origin, const Size32u& size, ExtraData&&) noexcept
            :GridItem(origin, size)
        {
        }
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
        int32_t bb_classifierBin = -1;
    };

    using UniformGridItem = GridItem<GridItemData>;
    using UniformGrid = GridPartition<UniformGridItem>;

    static_assert(sizeof(GridItemBase) == 4 * sizeof(uint32_t), "");

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
