#ifndef FRAC_TRANSFORM_H_
#define FRAC_TRANSFORM_H_

#include "image/image.h"
#include "utils/size.hpp"
#include "utils/point2d.hpp"
#include "image/sampler.h"
#include <iostream>

namespace Frac {
	class Transform {
	public:
		enum Type {
			Id,
			Rotate_90,
			Rotate_180,
			Rotate_270,
			Flip,
			Flip_Rotate_90,
			Flip_Rotate_180,
			Flip_Rotate_270
		};
		enum Interpolation {
			NearestNeighbor,
			Bilinear
		};

		explicit Transform(Type t = Id) noexcept
			: _type(t)
		{

		}
		void setType(const Type t) noexcept {
			this->_type = t;
		}
		Type type() const noexcept {
			return _type;
		}
		Type next() {
			if (_type == Rotate_270) {
				_type = Id;
			} else {
				_type = static_cast<Type>((int)(_type + 1));
			}
			return _type;
		}
		Transform inverse() const {
			switch (_type) {
			case Rotate_90:
				return Transform(Rotate_270);
			case Rotate_270:
				return Transform(Rotate_90);
			case Flip_Rotate_90:
				return Transform(Flip_Rotate_270);
			case Flip_Rotate_270:
				return Transform(Flip_Rotate_90);
			default:
				return *this;
			}
		}
		Image resize(const Image& source, const Size32u& targetSize, Interpolation t = NearestNeighbor) const {
			switch(t) {
			case Bilinear:
				return this->_resize_b(source, targetSize);
			default:
				return this->_resize_nn(source, targetSize);
			}
		}
		Image map(const Image& source) const {
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
		Size32u map(const Size32u& s) const noexcept {
			switch(_type) {
			case Rotate_90:
			case Rotate_270:
			case Flip_Rotate_90:
			case Flip_Rotate_270:
				return Size32u(s.y(), s.x());
			default:
				return s;
			}
		}
		template <typename T>
		Point2d<T> map(const T x, const T y, const Size32u& s) const noexcept {
			static const int __map_lookup[8][8] = {
		/*ID*/		{ 1, 0, 0, 0,  0, 1, 0, 0 },
		/*90*/		{0, 1, 0, 0,  -1, 0, 1, 0},
		/*180*/		{-1, 0, 1, 0,  0, -1, 0, 1},
		/*270*/		{0, -1, 0, 1,  1, 0, 0, 0},
		/*flip*/	{1, 0, 0, 0,   0, -1, 0, 1},
		/*fl 90*/	{0, 1, 0, 0,   1, 0, 0, 0},
		/*fl 180*/	{-1, 0, 1, 0,  0, 1, 0, 0},
		/*fl 270*/	{0, -1, 0, 1, -1, 0, 1, 0}
			};
			return Point2d<T>(__map_lookup[_type][0] * x + __map_lookup[_type][1] * y + __map_lookup[_type][2] * (s.x() - 1) + __map_lookup[_type][3] * (s.y() - 1),
				__map_lookup[_type][4] * x + __map_lookup[_type][5] * y + __map_lookup[_type][6] * (s.x() - 1) + __map_lookup[_type][7] * (s.y() - 1));
		}
		void copy(const Image& source, Image& target, const double contrast = 1.0, const double brightness = 0.0) const;
	private:
		Image _resize_nn(const Image& source, const Size32u& targetSize) const;
		Image _resize_b(const Image& source, const Size32u& targetSize) const;
		template <typename Sampler>
		Image _resize_impl(const Image& source, const Size32u& targetSize) const {
			if (source.size() != targetSize || this->_type != Id) {
				const Sampler samplerSource(source);
				AbstractBufferPtr<Image::Pixel> buffer = Buffer<Image::Pixel>::alloc(targetSize.x() * targetSize.y());
				auto* targetPtr = buffer->get();
				for (uint32_t y = 0 ; y<targetSize.y() ; ++y) {
					for (uint32_t x = 0 ; x<targetSize.x() ; ++x) {
						const uint32_t srcY = (y * source.height()) / targetSize.y();
						const uint32_t srcX = (x * source.width()) / targetSize.x();
						const auto p = this->map(srcX, srcY, source.size());
						targetPtr[x + y * targetSize.x()] = samplerSource(p.x(), p.y());
					}
				}
				return Image(buffer, targetSize.x(), targetSize.y(), targetSize.x());
			} else {
				return source;
			}
		}
	private:
		Type _type = Id;
	};
}

#endif
