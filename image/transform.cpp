#include "image/transform.h"
#include "image/image.h"
#include "image/sampler.h"

using namespace Frac;

void Transform::copy(const Image& source, Image& target, const double contrast, const double brightness) const {
	const auto targetSize = target.size();
	const auto targetPtr = target.data()->get();
	const SamplerBilinear sourceSampler(source);
	for (uint32_t y = 0; y < target.height(); ++y)
		for (uint32_t x = 0; x < target.width(); ++x) {
		const uint32_t srcY = (y * source.height()) / targetSize.y();
		const uint32_t srcX = (x * source.width()) / targetSize.x();
		const double result = contrast * convert<double, Image::Pixel>(sourceSampler(srcX, srcY, *this, source.size())) + brightness;
		targetPtr[x + y * target.stride()] = result < 0.0 ? 0 : result > 255 ? 255 : (Image::Pixel)(result);
	}
}

template <typename Sampler>
Image _resize_impl(const Transform& t, const Image& source, const Size32u& targetSize) {
	if (source.size() != targetSize || t.type() != Transform::Id) {
		const Sampler samplerSource(source);
		AbstractBufferPtr<Image::Pixel> buffer = Buffer<Image::Pixel>::alloc(targetSize.x() * targetSize.y());
		auto* targetPtr = buffer->get();
		for (uint32_t y = 0; y<targetSize.y(); ++y) {
			for (uint32_t x = 0; x<targetSize.x(); ++x) {
				const uint32_t srcY = (y * source.height()) / targetSize.y();
				const uint32_t srcX = (x * source.width()) / targetSize.x();
				const auto p = t.map(srcX, srcY, source.size());
				targetPtr[x + y * targetSize.x()] = samplerSource(p.x(), p.y());
			}
		}
		return Image(buffer, targetSize.x(), targetSize.y(), targetSize.x());
	}
	else {
		return source;
	}
}

Image Transform::_resize_nn(const Image& source, const Size32u& targetSize) const {
	return _resize_impl<SamplerLinear>(*this, source, targetSize);
}

Image Transform::_resize_b(const Image& source, const Size32u& targetSize) const {
	return _resize_impl<SamplerBilinear>(*this, source, targetSize);
}

Image Transform::map(const Image& source) const {
	if (this->_type == Id)
		return source;
	const Size32u targetSize = this->map(source.size());
	AbstractBufferPtr<Image::Pixel> buffer = Buffer<Image::Pixel>::alloc(targetSize.x() * targetSize.y());
	const auto* sourcePtr = source.data()->get();
	auto* targetPtr = buffer->get();
	Image result(buffer, targetSize.x(), targetSize.y(), targetSize.x());
	for (uint32_t y = 0; y < result.height(); ++y)
		for (uint32_t x = 0; x < result.width(); ++x) {
			const Point2du p = this->map(x, y, targetSize);
			targetPtr[x + y * targetSize.x()] = sourcePtr[p.x() + p.y() * source.stride()];
		}
	return result;
}

Image Transform::resize(const Image& source, const Size32u& targetSize, Interpolation t) const {
	switch (t) {
	case Bilinear:
		return this->_resize_b(source, targetSize);
	default:
		return this->_resize_nn(source, targetSize);
	}
}
