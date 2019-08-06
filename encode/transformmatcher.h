#ifndef TRANSFORMMATCHER_H
#define TRANSFORMMATCHER_H

#include "image/transform.h"
#include "image/metrics.h"
#include "encode/datatypes.h"
#include <iostream>

#include "image/partition2.hpp"
#include "image/Image2.hpp"
#include "image/ImageStatistics.hpp"

#ifdef FRAC_WITH_AVX
#include "utils/sse_utils.h"
#include <immintrin.h>
#endif

namespace Frac {
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

class TransformMatcher {
public:
	TransformMatcher(const Metric& metric, const double rmsThreshold, const double sMax)
		:_metric(metric)
		,_rmsThreshold(rmsThreshold)
		,_sMax(sMax)
	{

	}
	double truncateSMax(const double s) const noexcept {
		if (_sMax > 0.0)
			return s > _sMax ? _sMax : (s < -_sMax ? -_sMax : s);
		return s;
	}
	bool checkDistance(const double d) const noexcept {
		return d <= _rmsThreshold;
	}

    // Frac2

	transform_score_t match(const Frac2::ImagePlane& source, const Frac2::UniformGridItem& sourcePatch,
		const Frac2::ImagePlane& target, const Frac2::UniformGridItem& targetPatch) const
	{
		if (sourcePatch.size.x() == 16 && sourcePatch.size.y() == 16 && targetPatch.size.x() == 4 && targetPatch.size.y() == 4)
			return this->match_16to4(source, sourcePatch, target, targetPatch);
		return this->match_generic(source, sourcePatch, target, targetPatch);
	}

    transform_score_t match_generic(const Frac2::ImagePlane& source, const Frac2::UniformGridItem& sourcePatch,
        const Frac2::ImagePlane& target, const Frac2::UniformGridItem& targetPatch) const {
        transform_score_t result;
        Transform t(Transform::Id);
		const double sumA = Frac2::ImageStatistics2::sum(target, targetPatch);
        do {
            transform_score_t candidate;
            candidate.distance = this->_metric.distance(source, target, sourcePatch, targetPatch, t);
            candidate.transform = t.type();
            if (candidate.distance <= result.distance) {
                const double N = targetPatch.size.area();
                double sumA2 = 0.0, sumB = 0.0, sumAB = 0.0;
                for (uint32_t y = 0; y < targetPatch.size.y(); ++y) {
                    for (uint32_t x = 0; x < targetPatch.size.x(); ++x) {
                        const auto srcY = (y * sourcePatch.size.y()) / targetPatch.size.y();
                        const auto srcX = (x * sourcePatch.size.x()) / targetPatch.size.x();
                        const double valA = target.value<double>(targetPatch.origin.x() + x, targetPatch.origin.y() + y);
                        const double valB = SamplerBilinear::sample<double>(source, sourcePatch, srcX, srcY, t);
                        sumB += valB;
                        sumA2 += valA * valA;
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
        return result;
    }

	transform_score_t match_16to4(const Frac2::ImagePlane& source, const Frac2::UniformGridItem& sourcePatch,
		const Frac2::ImagePlane& target, const Frac2::UniformGridItem& targetPatch) const {
		transform_score_t result;
		Transform t(Transform::Id);
		const double sumA = Frac2::ImageStatistics2::sum(target, targetPatch);
		do {
			transform_score_t candidate;
			candidate.distance = this->_metric.distance(source, target, sourcePatch, targetPatch, t);
			candidate.transform = t.type();
			if (candidate.distance <= result.distance) {
				const double N = targetPatch.size.area();
				double sumA2 = 0.0, sumB = 0.0, sumAB = 0.0;
				for (uint32_t y = 0; y < 4; ++y) {
					const auto srcY = y * 4;
					for (uint32_t x = 0; x < 4; ++x) {
						const auto srcX = x * 4;
						const double valA = target.value<double>(targetPatch.origin.x() + x, targetPatch.origin.y() + y);
						const double valB = SamplerBilinear::sample<double>(source, sourcePatch, srcX, srcY, t);
						sumB += valB;
						sumA2 += valA * valA;
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
		return result;
	}

private:
	const Metric& _metric;
	const double _rmsThreshold;
	const double _sMax;
};
}

#endif // TRANSFORMMATCHER_H
