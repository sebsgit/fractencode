#include "sampler.h"

using namespace Frac;

SamplerLinear::SamplerLinear(const Image& source)
    : _source(source.data()->get())
    , _stride(source.stride())
{

}
Image::Pixel SamplerLinear::operator() (uint32_t x, uint32_t y) const {
    return _source[x + y * _stride];
}

SamplerBilinear::SamplerBilinear(const Image& source)
    : _source(source.data()->get())
    , _stride(source.stride())
    , _width(source.width())
    , _height(source.height())
{

}

Image::Pixel SamplerBilinear::operator() (uint32_t x, uint32_t y) const {
    if (x == _width - 1)
        --x;
    if (y == _height - 1)
        --y;
    const int total = (int)_source[x + y * _stride] + (int)_source[x + 1 + y * _stride] + (int)_source[x + (y + 1) * _stride] + (int)_source[x + 1 + (y + 1) * _stride];
    return (Image::Pixel)(total / 4);
}
