#ifndef METRICS_H
#define METRICS_H

#include "image/image.h"
#include "image/transform.h"
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
            double sum = 0.0;
            const auto* srcA = a.data()->get();
            const auto* srcB = b.data()->get();
            if (a.size() == b.size()) {
                a.map([&](uint32_t x, uint32_t y) {
                    const Point2du p = t.map(x, y, a.size());
                    const double val = convert<double, Image::Pixel>(srcA[x + y * a.stride()]) - convert<double, Image::Pixel>(srcB[p.x() + p.y() * b.stride()]);
                    sum += val * val;
                });
            } else {
                const SamplerBilinear samplerB(b);
                a.map([&](uint32_t x, uint32_t y) {
                    const uint32_t yB = (y * b.height()) / a.height();
                    const uint32_t xB = (x * b.width()) / a.width();
                    const double val = convert<double, Image::Pixel>(srcA[x + y * a.stride()]) - convert<double, Image::Pixel>(samplerB(xB, yB, t, b.size()));
                    sum += val * val;
                });
            }
            return sqrt(sum / a.size().area());
        }
    };
}

#endif // METRICS_H
