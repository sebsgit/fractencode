#include "image/transform.h"

using namespace Frac;

void Transform::copy(const Image& source, Image& target, const double contrast, const double brightness) const {
    const auto targetSize = target.size();
    const auto targetPtr = target.data()->get();
    const SamplerBilinear sourceSampler(source);
    target.map([&](uint32_t x, uint32_t y) {
        const uint32_t srcY = (y * source.height()) / targetSize.y();
        const uint32_t srcX = (x * source.width()) / targetSize.x();
        const double result = contrast * convert<double, Image::Pixel>(sourceSampler(srcX, srcY, *this, source.size())) + brightness;
        targetPtr[x + y * target.stride()] = result < 0.0 ? 0 : result > 255 ? 255 : (uint8_t)result;
    });
}

Image Transform::_resize_nn(const Image& source, const Size32u& targetSize) const {
    return this->_resize_impl<SamplerLinear>(source, targetSize);
}

Image Transform::_resize_b(const Image& source, const Size32u& targetSize) const {
    return this->_resize_impl<SamplerBilinear>(source, targetSize);
}
