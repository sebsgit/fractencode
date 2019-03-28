#pragma once

#include "image/Image2.hpp"
#include "image/partition2.hpp"

namespace Frac2 {
    class ImageStatistics2 {
    public:
        static uint16_t sum_impl(const ImagePlane& image, const GridItemBase& item) noexcept;

        template <typename T = double>
        static T sum(const ImagePlane& image, const GridItemBase& item) noexcept {
            return static_cast<T>(sum_impl(image, item));
        }
        template <typename T = double>
        static T mean(const ImagePlane& image, const GridItemBase& item) noexcept {
            return sum<T>(image, item) / item.size.area();
        }
    };
}
