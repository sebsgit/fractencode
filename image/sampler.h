#ifndef SAMPLER_H
#define SAMPLER_H

#include "image.h"
#include "transform.h"

namespace Frac {

class Transform;

class SamplerLinear {
public:
	explicit SamplerLinear(const Image& source);
	Image::Pixel operator() (uint32_t x, uint32_t y) const;
private:
	const Image::Pixel* _source;
	const uint32_t _stride;
};

class SamplerBilinear {
public:
	explicit SamplerBilinear(const Image& source);
	Image::Pixel operator() (uint32_t x, uint32_t y) const;
	Image::Pixel operator() (uint32_t x, uint32_t y, const Transform& t, const Size32u& s) const;
private:
	const Image::Pixel* _source;
	const uint32_t _stride;
	const uint32_t _width;
	const uint32_t _height;
};
}

#endif // SAMPLER_H
