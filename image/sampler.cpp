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

Image::Pixel SamplerBilinear::operator() (uint32_t x, uint32_t y, const Transform& t, const Size32u& size) const {
	if (x == _width - 1)
		--x;
	if (y == _height - 1)
		--y;
	auto tl = t.map(x, y, size);
	auto tr = t.map(x + 1, y, size);
	auto bl = t.map(x, y + 1, size);
	auto br = t.map(x + 1, y + 1, size);
	const int total = (int)_source[tl.x() + tl.y() * _stride] + (int)_source[tr.x() + tr.y() * _stride] + (int)_source[bl.x() + bl.y() * _stride] + (int)_source[br.x() + br.y() * _stride];
	return (Image::Pixel)(total / 4);
}
