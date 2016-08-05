#ifndef SAMPLER_H
#define SAMPLER_H

#include "image.h"
#include "transform.h"

namespace Frac {
class SamplerLinear {
public:
    SamplerLinear(const Image& source);
    Image::Pixel operator() (uint32_t x, uint32_t y) const;
private:
    const Image::Pixel* _source;
    const uint32_t _stride;
};

class SamplerBilinear {
public:
    SamplerBilinear(const Image& source);
    Image::Pixel operator() (uint32_t x, uint32_t y) const;
private:
    const Image::Pixel* _source;
    const uint32_t _stride;
    const uint32_t _width;
    const uint32_t _height;
};
}

#endif // SAMPLER_H
