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
            //todo calculate variance & classify as texture / flat
            return true;
        }
    };
}

#endif // CLASSIFIER_H
