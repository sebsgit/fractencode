#include "Classifier2.hpp"
#include "image\ImageStatistics.hpp"

using namespace Frac2;

Classifier2::~Classifier2() = default;

int BrightnessBlocksClassifier2::getCategory(float a1, float a2, float a3, float a4) noexcept {
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

int BrightnessBlocksClassifier2::getCategory(const ImagePlane& image, const UniformGridItem& item)
{
    const float a1 = ImageStatistics2::mean(image, item.topLeft());
    const float a2 = ImageStatistics2::mean(image, item.topRight());
    const float a3 = ImageStatistics2::mean(image, item.bottomLeft());
    const float a4 = ImageStatistics2::mean(image, item.bottomRight());
    return BrightnessBlocksClassifier2::getCategory(a1, a2, a3, a4);
}

bool BrightnessBlocksClassifier2::compare(const UniformGridItem& item1, const UniformGridItem& item2) const
{
    int sourceCategory = -1;
    {
        auto key = this->cacheKey(item1);
        std::shared_lock readLock(this->_cacheLock);
        auto it = this->_cache.find(key);
        readLock.unlock();
        if (it == this->_cache.end()) {
            sourceCategory = getCategory(this->sourceImage(), item1);
            std::unique_lock writeLock(this->_cacheLock);
            this->_cache[key] = sourceCategory;
        }
        else {
            sourceCategory = it->second;
        }
    }
    return sourceCategory == getCategory(this->targetImage(), item2);
}
