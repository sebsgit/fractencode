#pragma once

#include "image/Image2.hpp"
#include "image/partition2.hpp"
#include "image/sampler.h"
#include "utils/buffer.hpp"

namespace Frac {
    static void copy(const Frac2::ImagePlane& source,
        Frac2::ImagePlane& target,
        const Frac2::GridItemBase& sourcePatch,
        const Frac2::GridItemBase& targetPatch,
        const double contrast,
        const double brightness,
        TransformType transform)
    {
        const auto targetPtr = target.data();
        for (uint32_t y = 0; y < targetPatch.size.y(); ++y)
            for (uint32_t x = 0; x < targetPatch.size.x(); ++x) {
                const uint32_t srcX = (x * sourcePatch.size.x()) / targetPatch.size.x();
                const uint32_t srcY = (y * sourcePatch.size.y()) / targetPatch.size.y();
                const double result = contrast * SamplerBilinear::sample<double>(source, sourcePatch, srcX, srcY, transform) + brightness;
                targetPtr[targetPatch.origin.x() + x + (targetPatch.origin.y() + y) * target.stride()] = result < 0.0 ? 0 : result > 255 ? 255 : convert<uint8_t>(result);
            }
    }
};

