#include "Classifier2.hpp"
#include "image\ImageStatistics.hpp"

using namespace Frac2;

Classifier2::~Classifier2() = default;

int BrightnessBlocksClassifier2::getCategory(double a1, double a2, double a3, double a4) noexcept {
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
    const auto a1 = ImageStatistics2::sum(image, item.topLeft());
    const auto a2 = ImageStatistics2::sum(image, item.topRight());
    const auto a3 = ImageStatistics2::sum(image, item.bottomLeft());
    const auto a4 = ImageStatistics2::sum(image, item.bottomRight());
    return BrightnessBlocksClassifier2::getCategory(a1, a2, a3, a4);
}

bool BrightnessBlocksClassifier2::compare(const UniformGridItem& item1, const UniformGridItem& item2) const
{
    int sourceCategory = -1;
    {
        /*TODO: make this faster, precompute and dont use std::unordered_map

        auto key = this->cacheKey(item1);
        std::shared_lock<std::shared_mutex> readLock(this->_cacheLock);
        auto it = this->_cache.find(key);
        readLock.unlock();
        if (it == this->_cache.end()) {
            sourceCategory = getCategory(this->sourceImage(), item1);
            std::unique_lock<std::shared_mutex> writeLock(this->_cacheLock);
            this->_cache[key] = sourceCategory;
        }
        else {
            sourceCategory = it->second;
        } */
        sourceCategory = getCategory(this->sourceImage(), item1);
    }
    return sourceCategory == getCategory(this->targetImage(), item2);
}
