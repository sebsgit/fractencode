#ifndef METRICS_H
#define METRICS_H

#include "image/transform.h"
#include "image/sampler.h"
#include "image/partition2.hpp"
#include "image/Image2.hpp"
#include "utils/Assert.hpp"
#include "utils/buffer.hpp"
#include <cmath>

namespace Frac {

    template <TransformType transformType>
    class RootMeanSquare {
	public:
        /**
            Calculates the distance between two grid elements.
            @param t_a Transform to apply to element A.
        */
        double distance(const Frac2::ImagePlane& a, const Frac2::ImagePlane& b,
            const Frac2::GridItemBase& sliceA,
            const Frac2::GridItemBase& sliceB) const
        {
            constexpr auto t_a = Transform<transformType>();
            if (sliceA.size == sliceB.size) {
                int32_t sum = 0;
                for (uint32_t y = 0; y < sliceB.size.y(); ++y) {
                    for (uint32_t x = 0; x < sliceB.size.x(); ++x) {
                        const auto valB = b.value<int16_t>(sliceB.origin.x() + x, sliceB.origin.y() + y);
                        auto p = t_a.map(x, y, sliceA.origin.x(), sliceA.origin.y(), sliceA.size.x(), sliceA.size.y());
                        auto val = a.value<int16_t>(p) - valB;
                        sum += val * val;
                    }
                }
                return convert<double>(sum) / sliceA.size.area();
            } else {
                float sum = 0.0f;
                FRAC_ASSERT(sliceB.size.x() < sliceA.size.x());
                int16_t widthRatio = sliceA.size.x() / sliceB.size.x();
                int16_t heightRatio = sliceA.size.y() / sliceB.size.y();
                for (uint32_t y = 0; y < sliceB.size.y(); ++y) {
                    for (uint32_t x = 0; x < sliceB.size.x(); ++x) {
                        const auto valB = b.value<int16_t>(sliceB.origin.x() + x, sliceB.origin.y() + y);
                        auto val = valB - SamplerBilinear::sample<float>(a, sliceA, x * widthRatio, y * heightRatio, t_a);
                        sum += val * val;
                    }
                }
                return convert<double>(sum) / sliceA.size.area();
            }
        }

        double distance(const Frac2::ImagePlane& a, const Frac2::ImagePlane& b) const
        {
            return this->distance(a, b, Frac2::GridItemBase{Point2du(0, 0), a.size()}, Frac2::GridItemBase{Point2du(0, 0), b.size()});
        }
	};
}

#endif // METRICS_H
