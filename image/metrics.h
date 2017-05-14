#ifndef METRICS_H
#define METRICS_H

#include "image/image.h"
#include "image/transform.h"
#include "image/sampler.h"
#include <cmath>

namespace Frac {
	class Metric {
	public:
		virtual ~Metric() {}
		virtual double distance(const Image& a, const Image& b, const Transform& t = Transform()) const = 0;
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
	};
}

#endif // METRICS_H
