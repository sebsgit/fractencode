#ifndef METRICS_H
#define METRICS_H

#include "image/image.h"
#include "image/transform.h"
#include "image/sampler.h"
#include "image/partition2.hpp"
#include "image/Image2.hpp"
#include "utils/Assert.hpp"
#include <cmath>

namespace Frac {
	class Metric {
	public:
		virtual ~Metric() {}
		virtual double distance(const Image& a, const Image& b, const Transform& t = Transform()) const = 0;

        virtual double distance(const Frac2::ImagePlane& a, const Frac2::ImagePlane& b,
            const Frac2::GridItemBase& sliceA,
            const Frac2::GridItemBase& sliceB,
            const Transform& t = Transform()) const = 0;
	};

	class RootMeanSquare : public Metric {
	public:
		double distance(const Image& a, const Image& b, const Transform& t = Transform()) const override {
			int32_t sum = 0;
			const auto* srcA = a.data()->get();
			const auto* srcB = b.data()->get();
			if (a.width() == b.width() && a.height() == b.height()) {
				Point2du p;
				for (uint32_t y = 0; y < a.height(); ++y) {
					const auto y_off = y * a.stride();
					for (uint32_t x = 0; x < a.width(); ++x) {
						t.map(&p.x(), &p.y(), x, y, a.width(), a.height());
						const auto val = convert<int16_t>(srcA[x + y_off]) - convert<int16_t>(srcB[p.x() + p.y() * b.stride()]);
						sum += val * val;
					}
				}
			} else {
				const SamplerBilinear samplerB(b);
				for (uint32_t y = 0; y < a.height(); ++ y) {
					const auto y_off = y * a.stride();
					for (uint32_t x = 0; x < a.width(); ++x) {
						const uint32_t yB = (y * b.height()) / a.height();
						const uint32_t xB = (x * b.width()) / a.width();
						const auto val = convert<int16_t>(srcA[x + y_off]) - convert<int16_t>(samplerB(xB, yB, t, b.width(), b.height()));
						sum += val * val;
					}
				}
			}
			return sqrt(convert<double>(sum) / (a.width() * a.height()));
		}

        /**
            Calculates the distance between two grid elements.
            @param t_a Transform to apply to element A.
        */
        double distance(const Frac2::ImagePlane& a, const Frac2::ImagePlane& b,
            const Frac2::GridItemBase& sliceA,
            const Frac2::GridItemBase& sliceB,
            const Transform& t_a = Transform()) const override 
        {
            
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

        double distance(const Frac2::ImagePlane& a, const Frac2::ImagePlane& b,
            const Transform& t_a = Transform()) const
        {
            return this->distance(a, b, Frac2::GridItemBase{Point2du(0, 0), a.size()}, Frac2::GridItemBase{Point2du(0, 0), b.size()}, t_a);
        }
	};
}

#endif // METRICS_H
