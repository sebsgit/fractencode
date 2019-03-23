#pragma once

#include "image/Image2.hpp"
#include "image/partition2.hpp"

namespace Frac2 {
    class Classifier2 {
    public:
        Classifier2(const ImagePlane& source, const ImagePlane& target)
            :_sourceImage(source)
            , _targetImage(target)
        {}
        virtual ~Classifier2();

        const ImagePlane& sourceImage() const noexcept { return this->_sourceImage; }
        const ImagePlane& targetImage() const noexcept { return this->_targetImage; }

        virtual bool compare(const UniformGridItem& source, const UniformGridItem& target) const = 0;

    protected:
        const ImagePlane& _sourceImage;
        const ImagePlane& _targetImage;
    };

    class DummyClassifier : public Classifier2
    {
    public:
        using Classifier2::Classifier2;

        bool compare(const UniformGridItem&, const UniformGridItem&) const override {
            return true;
        }
    };
} // namespace Frac2
