#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "image.h"

namespace Frac {
    class ImageClassifier {
    public:
        virtual ~ImageClassifier() {}
        virtual bool compare(const Image& a, const Image& b) const = 0;
    public:
        double mean(const Image& image) const noexcept {
            auto& cache = image.cache();
            double result = cache.get(ImageData::KeyMean);
            if (result == -1.0) {
                double sum = 0.0;
                image.map([&](Image::Pixel p) { sum += p; });
                result = sum / image.size().area();
                cache.put(ImageData::KeyMean, result);
            }
            return result;
        }
        double variance(const Image& image) const noexcept {
            const double av = this->mean(image);
            double result = image.cache().get(ImageData::KeyVariance);
            if (result == -1.0) {
                double sum = 0.0;
                image.map([&](Image::Pixel p) { sum += (p - av) * (p - av); });
                result = sum / image.size().area();
                image.cache().put(ImageData::KeyVariance, result);
            }
            return result;
        }
    };

    class DummyClassifier : public ImageClassifier {
    public:
        ~DummyClassifier() {}
        bool compare(const Image&, const Image&) const override {
            return true;
        }
    };

    class TextureClassifier : public ImageClassifier {
    public:
        ~TextureClassifier() {}
        bool compare(const Image& a, const Image& b) const override {
            const auto va = this->variance(a);
            const auto vb = this->variance(b);
            return this->isFlat(va) == this->isFlat(vb);
        }
    protected:
        bool isFlat(const double var) const noexcept {
            return var < 5000.0;
        }
    };
}

#endif // CLASSIFIER_H
