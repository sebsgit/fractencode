#ifndef TRANSFORMMATCHER_H
#define TRANSFORMMATCHER_H

#include "image.h"
#include "transform.h"
#include "metrics.h"
#include "partition.h"
#include "datatypes.h"

namespace Frac {
class TransformMatcher {
public:
	TransformMatcher(const Metric& metric, const double rmsThreshold, const double sMax)
		:_metric(metric)
		,_rmsThreshold(rmsThreshold)
		,_sMax(sMax)
	{

	}

#ifdef FRAC_WITH_AVX
	static Image::Pixel sample_helper (uint32_t x, uint32_t y, const uint8_t* source,  const int type, const Size32u& size, const size_t stride) {
		static const int __map_lookup[8][8] = {
			/*ID*/{ 1, 0, 0, 0,  0, 1, 0, 0 },
			/*90*/{ 0, 1, 0, 0,  -1, 0, 1, 0 },
			/*180*/{ -1, 0, 1, 0,  0, -1, 0, 1 },
			/*270*/{ 0, -1, 0, 1,  1, 0, 0, 0 },
			/*flip*/{ 1, 0, 0, 0,   0, -1, 0, 1 },
			/*fl 90*/{ 0, 1, 0, 0,   1, 0, 0, 0 },
			/*fl 180*/{ -1, 0, 1, 0,  0, 1, 0, 0 },
			/*fl 270*/{ 0, -1, 0, 1, -1, 0, 1, 0 }
		};

		if (x == size.x() - 1)
			--x;
		if (y == size.y() - 1)
			--y;

		auto tl = Point2d<uint32_t>(__map_lookup[type][0] * x + __map_lookup[type][1] * y + __map_lookup[type][2] * (size.x() - 1) + __map_lookup[type][3] * (size.y() - 1),
			__map_lookup[type][4] * x + __map_lookup[type][5] * y + __map_lookup[type][6] * (size.x() - 1) + __map_lookup[type][7] * (size.y() - 1));

		auto tr = Point2d<uint32_t>(__map_lookup[type][0] * (x + 1) + __map_lookup[type][1] * y + __map_lookup[type][2] * (size.x() - 1) + __map_lookup[type][3] * (size.y() - 1),
			__map_lookup[type][4] * (x + 1) + __map_lookup[type][5] * y + __map_lookup[type][6] * (size.x() - 1) + __map_lookup[type][7] * (size.y() - 1));

		auto bl = Point2d<uint32_t>(__map_lookup[type][0] * x + __map_lookup[type][1] * (y + 1) + __map_lookup[type][2] * (size.x() - 1) + __map_lookup[type][3] * (size.y() - 1),
			__map_lookup[type][4] * x + __map_lookup[type][5] * (y + 1) + __map_lookup[type][6] * (size.x() - 1) + __map_lookup[type][7] * (size.y() - 1));

		auto br = Point2d<uint32_t>(__map_lookup[type][0] * (x + 1) + __map_lookup[type][1] * (y + 1) + __map_lookup[type][2] * (size.x() - 1) + __map_lookup[type][3] * (size.y() - 1),
			__map_lookup[type][4] * (x + 1) + __map_lookup[type][5] * (y + 1) + __map_lookup[type][6] * (size.x() - 1) + __map_lookup[type][7] * (size.y() - 1));

		const int total = (int)source[tl.x() + tl.y() * stride] + (int)source[tr.x() + tr.y() * stride] + (int)source[bl.x() + bl.y() * stride] + (int)source[br.x() + br.y() * stride];
		return (Image::Pixel)(total / 4);
	}
#endif

	transform_score_t match(const PartitionItemPtr& a, const PartitionItemPtr& b) const {
		transform_score_t result;
#ifdef FRAC_WITH_AVX
		Transform t(Transform::Id);
		do {
			transform_score_t candidate;
			candidate.distance = _metric.distance(a->image(), b->image(), t);
			candidate.transform = t.type();
			if (candidate.distance <= result.distance) {
				const double N = (double)(a->image().width()) * a->image().height();
				double sumA = 0.0, sumA2 = 0.0, sumB = 0.0, sumB2 = 0.0, sumAB = 0.0;
				for (uint32_t y = 0; y<a->image().height(); ++y) {
					const auto srcY = (y * b->image().height()) / a->image().height();
					for (uint32_t x = 0; x<a->image().width(); ++x) {
						const auto srcX = (x * b->image().width()) / a->image().width();
						const double valA = convert<double>(a->image().data()->get()[x + y * a->image().stride()]);
						const double valB = convert<double>(sample_helper(srcX, srcY, b->image().data()->get(), t.type(), b->image().size(), b->image().stride()));
						sumA += valA;
						sumB += valB;
						sumA2 += valA * valA;
						sumB2 += valB * valB;
						sumAB += valA * valB;
					}
				}
				const double tmp = (N * sumA2 - (sumA - 1) * sumA);
				const double s = this->truncateSMax(fabs(tmp) < 0.00001 ? 0.0 : (N * sumAB - sumA * sumB) / tmp);
				const double o = (sumB - s * sumA) / N;
				candidate.contrast = s;
				candidate.brightness = o;
				result = candidate;
			}
			if (this->checkDistance(result.distance))
				break;
		} while (t.next() != Transform::Id);
#else
		Transform t(Transform::Id);
		const SamplerBilinear samplerB(b->image());
		do {
			transform_score_t candidate;
			candidate.distance = _metric.distance(a->image(), b->image(), t);
			candidate.transform = t.type();
			if (candidate.distance <= result.distance) {
				const double N = (double)(a->image().width()) * a->image().height();
				double sumA = ImageStatistics::sum(a->image()), sumA2 = 0.0, sumB = 0.0, sumB2 = 0.0, sumAB = 0.0;
				for (uint32_t y = 0 ; y<a->image().height() ; ++y) {
					for (uint32_t x = 0 ; x<a->image().width() ; ++x) {
						const auto srcY = (y * b->image().height()) / a->image().height();
						const auto srcX = (x * b->image().width()) / a->image().width();
						const double valA = convert<double>(a->image().data()->get()[x + y * a->image().stride()]);
						const double valB = convert<double>(samplerB(srcX, srcY, t, b->image().size()));
						sumB += valB;
						sumA2 += valA * valA;
						sumB2 += valB * valB;
						sumAB += valA * valB;
					}
				}
				const double tmp = (N * sumA2 - (sumA - 1) * sumA);
				const double s = this->truncateSMax( fabs(tmp) < 0.00001 ? 0.0 : (N * sumAB - sumA * sumB) / tmp );
				const double o = (sumB - s * sumA) / N;
				candidate.contrast = s;
				candidate.brightness = o;
				result = candidate;
			}
			if (this->checkDistance(result.distance))
				break;
		} while (t.next() != Transform::Id);
#endif
		return result;
	}
	double truncateSMax(const double s) const noexcept {
		if (_sMax > 0.0)
			return s > _sMax ? _sMax : (s < -_sMax ? -_sMax : s);
		return s;
	}
	bool checkDistance(const double d) const noexcept {
		return d <= _rmsThreshold;
	}
private:
	const Metric& _metric;
	const double _rmsThreshold;
	const double _sMax;
};
}

#endif // TRANSFORMMATCHER_H
