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
        CombinedClassifier() {

        }
        CombinedClassifier(ImageClassifier* c1) {
            _classifiers.push_back(std::shared_ptr<ImageClassifier>(c1));
        }
        CombinedClassifier(ImageClassifier* c1, ImageClassifier* c2) {
            _classifiers.push_back(std::shared_ptr<ImageClassifier>(c1));
            _classifiers.push_back(std::shared_ptr<ImageClassifier>(c2));
        }
        CombinedClassifier(ImageClassifier* c1, ImageClassifier* c2, ImageClassifier* c3) {
            _classifiers.push_back(std::shared_ptr<ImageClassifier>(c1));
            _classifiers.push_back(std::shared_ptr<ImageClassifier>(c2));
            _classifiers.push_back(std::shared_ptr<ImageClassifier>(c3));
        }
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
            return var < 2000 ? 0 : var < 4000 ? 1 : var < 6000 ? 2 : 3;
        }
    };

    class BrightnessBlockClassifier : public ImageClassifier {
    public:
        bool compare(const Image& a, const Image& b) const override {
            int typeA = a.cache().get(ImageData::KeyBlockTypeBrightness, -1);
            int typeB = b.cache().get(ImageData::KeyBlockTypeBrightness, -1);
            if (typeA == -1) {
                typeA = getCategory(a);
                a.cache().put(ImageData::KeyBlockTypeBrightness, typeA);
            }
            if (typeB == -1) {
                typeB = getCategory(b);
                b.cache().put(ImageData::KeyBlockTypeBrightness, typeB);
            }
            return typeA == typeB;
        }
    private:
        static int getCategory(const Image& image) {
            const auto halfW = image.width() / 2;
            const auto halfH = image.height() / 2;
            const double a1 = ImageStatistics::mean(image.slice(0, 0, halfW, halfH));
            const double a2 = ImageStatistics::mean(image.slice(halfW, 0, halfW, halfH));
            const double a3 = ImageStatistics::mean(image.slice(0, halfH, halfW, halfH));
            const double a4 = ImageStatistics::mean(image.slice(halfW, halfH, halfW, halfH));
            return category(a1, a2, a3, a4);
        }
        static int category(const double a1, const double a2, const double a3, const double a4) noexcept {
            const bool a1a2 = a1 > a2;
            const bool a1a3 = a1 > a3;
            const bool a1a4 = a1 > a4;
            const bool a2a1 = a2 > a1;
            const bool a2a3 = a2 > a3;
            const bool a2a4 = a2 > a4;
            const bool a3a1 = a3 > a1;
            const bool a3a2 = a3 > a2;
            const bool a3a4 = a3 > a4;
            const bool a4a1 = a4 > a1;
            const bool a4a2 = a4 > a2;
            const bool a4a3 = a4 > a3;

            if (a1a2 && a2a3 && a3a4) return 0;
            if (a3a1 && a1a4 && a4a2) return 0;
            if (a4a3 && a3a2 && a2a1) return 0;
            if (a2a4 && a4a1 && a1a3) return 0;

            if (a1a3 && a3a2 && a2a4) return 1;
            if (a2a1 && a1a4 && a4a3) return 1;
            if (a4a2 && a2a3 && a3a1) return 1;
            if (a3a4 && a4a1 && a1a2) return 1;

            if (a1a4 && a4a3 && a3a2) return 2;
            if (a4a1 && a1a2 && a2a3) return 2;
            if (a3a2 && a2a4 && a4a1) return 2;
            if (a2a3 && a3a1 && a1a4) return 2;

            if (a1a2 && a2a4 && a4a3) return 3;
            if (a3a1 && a1a2 && a2a4) return 3;
            if (a4a3 && a3a1 && a1a2) return 3;
            if (a2a4 && a4a3 && a3a1) return 3;

            if (a2a1 && a1a3 && a3a4) return 4;
            if (a1a3 && a3a4 && a4a2) return 4;
            if (a3a4 && a4a2 && a2a1) return 4;
            if (a4a2 && a2a1 && a1a3) return 4;

            if (a1a4 && a4a2 && a2a3) return 5;
            if (a4a1 && a1a3 && a3a4) return 5;
            if (a2a3 && a3a4 && a4a1) return 5;
            if (a3a2 && a2a1 && a1a4) return 5;

            return -1;
        }
    };
}

#endif // CLASSIFIER_H
