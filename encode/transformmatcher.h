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
    transform_score_t match(const PartitionItemPtr& a, const PartitionItemPtr& b) const {
        transform_score_t result;
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
