#ifndef FRAC_TRANSFORM_H_
#define FRAC_TRANSFORM_H_

#include "utils/size.hpp"
#include "utils/point2d.hpp"
#include "gpu/cuda/CudaConf.h"
#include <iostream>

namespace Frac {
	class Image;

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

		CUDA_CALLABLE explicit Transform(Type t = Id) noexcept
			: _type(t)
		{

		}
		CUDA_CALLABLE ~Transform() { }
		CUDA_CALLABLE void setType(const Type t) noexcept {
			this->_type = t;
		}
		CUDA_CALLABLE Type type() const noexcept {
			return _type;
		}
		CUDA_CALLABLE Type next() {
			if (_type == Rotate_270) {
				_type = Id;
			} else {
				_type = static_cast<Type>((int)(_type + 1));
			}
			return _type;
		}
		CUDA_CALLABLE Transform inverse() const {
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
		Image resize(const Image& source, const Size32u& targetSize, Interpolation t = NearestNeighbor) const;
		Image map(const Image& source) const;
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
			Point2d<T> result;
			this->map(&result.x(), &result.y(), x, y, s.x(), s.y());
			return result;
		}
		template <typename T>
		Point2d<T> map(const T x, const T y, const uint32_t w, const uint32_t h) const noexcept {
			Point2d<T> result;
			this->map(&result.x(), &result.y(), x, y, w, h);
			return result;
		}
		template <typename T> CUDA_CALLABLE
		void map(T* rx, T* ry, const T x, const T y, const T sx, const T sy) const noexcept {
			static const int __map_lookup[8][8] = {
				/*ID*/{ 1, 0, 0, 0,  0, 1, 0, 0 },
				/*90*/{ 0, 1, 0, 0,  -1, 0, 1, 0 },
				/*180*/{ -1, 0, 1, 0,  0, -1, 0, 1 },
				/*270*/{ 0, -1, 0, 1,  1, 0, 0, 0 },
				/*flip*/{ 1, 0, 0, 0,   0, -1, 0, 1 },
				/*fl 90*/{ 0, 1, 0, 0,   1, 0, 0, 0 },
				/*fl 180*/{ -1, 0, 1, 0,  0, 1, 0, 0 },
				/*fl 270*/{ 0, -1, 0, 1, -1, 0, 1, 0 }
			};
			*rx = __map_lookup[_type][0] * x + __map_lookup[_type][1] * y + __map_lookup[_type][2] * (sx - 1) + __map_lookup[_type][3] * (sy - 1);
			*ry = __map_lookup[_type][4] * x + __map_lookup[_type][5] * y + __map_lookup[_type][6] * (sx - 1) + __map_lookup[_type][7] * (sy - 1);
		}
		void copy(const Image& source, Image& target, const double contrast = 1.0, const double brightness = 0.0) const;
	private:
		Image _resize_nn(const Image& source, const Size32u& targetSize) const;
		Image _resize_b(const Image& source, const Size32u& targetSize) const;
	private:
		Type _type = Id;
	};
}

#endif
