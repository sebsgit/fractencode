#include "image.h"

using namespace Frac;

double ImageStatistics::sum(const Image& a) noexcept {
    double result = a.cache().get(ImageData::KeySum);
    if (result == -1.0) {
        a.map([&](Image::Pixel v) { result += v; });
        a.cache().put(ImageData::KeySum, result);
    }
    return result;
}
double ImageStatistics::mean(const Image& image) noexcept {
    auto& cache = image.cache();
    double result = cache.get(ImageData::KeyMean);
    if (result == -1.0) {
        result = sum(image) / image.size().area();
        cache.put(ImageData::KeyMean, result);
    }
    return result;
}
double ImageStatistics::variance(const Image& image) noexcept {
    const double av = mean(image);
    double result = image.cache().get(ImageData::KeyVariance);
    if (result == -1.0) {
        double sum = 0.0;
        image.map([&](Image::Pixel p) { sum += (p - av) * (p - av); });
        result = sum / image.size().area();
        image.cache().put(ImageData::KeyVariance, result);
    }
    return result;
}
