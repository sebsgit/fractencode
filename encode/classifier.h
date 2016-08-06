#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "image.h"

namespace Frac {
    class ImageClassifier {
    public:
        virtual ~ImageClassifier() {}
        virtual bool compare(const Image& a, const Image& b) const = 0;
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
            const auto va = ImageStatistics::variance(a);
            const auto vb = ImageStatistics::variance(b);
            return this->isFlat(va) == this->isFlat(vb);
        }
    protected:
        bool isFlat(const double var) const noexcept {
            return var < 2500.0;
        }
    };
}

#endif // CLASSIFIER_H
