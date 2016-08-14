#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "image.h"
#include <memory>
#include <vector>

namespace Frac {
    class ImageClassifier {
    public:
        virtual ~ImageClassifier() {}
        virtual bool compare(const Image& a, const Image& b) const = 0;
    };

    class CombinedClassifier : public ImageClassifier {
    public:
        CombinedClassifier& add(std::shared_ptr<ImageClassifier> p) {
            this->_classifiers.push_back(p);
            return *this;
        }
        bool compare(const Image& a, const Image& b) const override {
            for (const auto& p : _classifiers)
                if (!p->compare(a, b))
                    return false;
            return true;
        }
    private:
        std::vector<std::shared_ptr<ImageClassifier>> _classifiers;
    };

    class DummyClassifier : public ImageClassifier {
    public:
        bool compare(const Image&, const Image&) const override {
            return true;
        }
    };

    class TextureClassifier : public ImageClassifier {
    public:
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

    class ThresholdClassifier : public ImageClassifier {
    public:
        bool compare(const Image& a, const Image& b) const override {
            const auto va = ImageStatistics::variance(a);
            const auto vb = ImageStatistics::variance(b);
            return this->category(va) == this->category(vb);
        }
    protected:
        int category(const double var) const noexcept {
            return var < 2500 ? 0 : var < 5000 ? 1 : var < 7500 ? 2 : 3;
        }
    };
}

#endif // CLASSIFIER_H
