#pragma once

#include "image/Image2.hpp"
#include "image/partition2.hpp"

#include <shared_mutex>

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
        inline static uint64_t cacheKey(const GridItemBase& it) noexcept
        {
            return (it.origin.x() << 32) + it.origin.y();
        }

    protected:
        const ImagePlane& _sourceImage;
        const ImagePlane& _targetImage;

        std::shared_mutex _cacheLock;
        std::unordered_map<uint64_t, float> _cache;
    };

    class DummyClassifier : public Classifier2
    {
    public:
        using Classifier2::Classifier2;

        bool compare(const UniformGridItem&, const UniformGridItem&) const override {
            return true;
        }
    };

    class BrightnessBlocksClassifier2 : public Classifier2
    {
    public:
        using Classifier2::Classifier2;

        bool compare(const UniformGridItem& item1, const UniformGridItem& item2) const override;

    private:
        static int getCategory(const ImagePlane& image, const UniformGridItem& item);
        static int getCategory(float a1, float a2, float a3, float a4) noexcept;
    };
} // namespace Frac2
