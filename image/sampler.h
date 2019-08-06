#ifndef SAMPLER_H
#define SAMPLER_H

#include "Image2.hpp"
#include "partition2.hpp"
#include "transform.h"

namespace Frac {

class Transform;

class SamplerBilinear {
public:
    /**
        Interpolate a pixel value.
        @param image Image to read pixels from.
        @param patch Sampled region of interest
        @param x Local X coordinate within the sampled patch.
        @param y Local Y coodtinate within the sampled patch.
        @param t Transform to apply to the pixel coordinates within the sampled patch.
        @return Interpolated pixel value.
    */
    template <typename T>
    static T sample(
        const Frac2::ImagePlane& image,
        const Frac2::GridItemBase& patch,
        uint32_t x,
        uint32_t y,
        const Transform& t
        )
    {
        FRAC_ASSERT(x >= 0 && x <patch.size.x());
        FRAC_ASSERT(y >= 0 && y < patch.size.y());
        if (x == patch.size.x() - 1)
            --x;
        if (y == patch.size.y() - 1)
            --y;
        const auto offsets = t.generateSampleOffsets(image.stride(), x, y, patch.origin, patch.size);
        return image.sumAt<uint16_t>(offsets) / static_cast<T>(4);
    }
private:
	const uint32_t _stride;
	const uint32_t _width;
	const uint32_t _height;
};
}

#endif // SAMPLER_H
