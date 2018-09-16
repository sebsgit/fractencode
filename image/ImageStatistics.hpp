#pragma once

#include "image/Image2.hpp"
#include "image/partition2.hpp"

namespace Frac2 {
    class ImageStatistics2 {
    public:
        static double sum(const ImagePlane& image, const GridItemBase& item) noexcept {
            double result = 0.0;
            for (uint32_t y = 0; y < item.size.y(); ++y)
                for (uint32_t x = 0; x < item.size.x(); ++x)
                    result += image.value(item.origin.x() + x, item.origin.y() + y);
            return result;
        }
    };
}
