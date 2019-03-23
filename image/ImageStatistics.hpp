#pragma once

#include "image/Image2.hpp"
#include "image/partition2.hpp"

namespace Frac2 {
    class ImageStatistics2 {
    public:
        template <typename T = double>
        static T sum(const ImagePlane& image, const GridItemBase& item) noexcept {
            T result = T{ 0 };
            for (uint32_t y = 0; y < item.size.y(); ++y)
                for (uint32_t x = 0; x < item.size.x(); ++x)
                    result += image.value(item.origin.x() + x, item.origin.y() + y);
            return result;
        }
        template <typename T = double>
        static T sum(const ImagePlane& image, const UniformGridItem& item) noexcept {
            //TODO: maybe cache
            T result = T{0};
            for (uint32_t y = 0; y < item.size.y(); ++y)
                for (uint32_t x = 0; x < item.size.x(); ++x)
                    result += image.value(item.origin.x() + x, item.origin.y() + y);
            return result;
        }
        template <typename T = double>
        static T mean(const ImagePlane& image, const UniformGridItem& item) noexcept {
            return sum<T>(image, item) / item.size.area();
        }
        template <typename T = double>
        static T mean(const ImagePlane& image, const GridItemBase& item) noexcept {
            return sum<T>(image, item) / item.size.area();
        }
    };
}
