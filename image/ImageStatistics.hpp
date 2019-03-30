#pragma once

#include "image/Image2.hpp"
#include "image/partition2.hpp"

namespace Frac2 {
    class ImageStatistics2 {
    public:
        static uint16_t sum_u16(const ImagePlane& image, const GridItemBase& item) noexcept;
        static uint32_t sum_u32(const ImagePlane& image, const GridItemBase& item) noexcept;

        template <typename T = double>
        static T sum(const ImagePlane& image, const GridItemBase& item) noexcept {
            if (item.size.x() > 16)
                return static_cast<T>(sum_u32(image, item));
            return static_cast<T>(sum_u16(image, item));
        }
        template <typename T = double>
        static T mean(const ImagePlane& image, const GridItemBase& item) noexcept {
            return sum<T>(image, item) / item.size.area();
        }
    };
}
