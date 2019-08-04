#ifndef FRAC_TRANSFORM_H_
#define FRAC_TRANSFORM_H_

#include "utils/size.hpp"
#include "utils/point2d.hpp"
#include "gpu/cuda/CudaConf.h"
#include "utils/Assert.hpp"
#include <iostream>

namespace Frac2 {
    class ImagePlane;
    class GridItemBase;
};

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

		CUDA_CALLABLE Transform(Type t = Id) noexcept
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
		Point2d<T> map(const T x, const T y, const uint32_t w, const uint32_t h) const noexcept {
			Point2d<T> result;
			this->map(&result.x(), &result.y(), x, y, w, h);
			return result;
		}
        /**
            Transform the local patch coordinates.
            @param local_x Local X coordinate to transform.
            @param local_y Local Y coordinate to transform.
            @param patch_offset_x Global X position of the patch (in image coordinates).
            @param patch_offset_y Global Y position of the patch (in image coordinates).
            @param patch_width Width of the patch.
            @param patch_height Height of the patch.
            @return Coordinates of the transformed point in the global image coordinate system.
        */
        template <typename T>
        auto map(T local_x, T local_y, T patch_offset_x, T patch_offset_y, T patch_width, T patch_height) const noexcept {
            FRAC_ASSERT(local_x >= 0 && local_x < patch_width);
            FRAC_ASSERT(local_y >= 0 && local_y < patch_height);
			T x, y;
			this->map(&x, &y, local_x, local_y, patch_width, patch_height);
            return Point2d<T>(x + patch_offset_x, y + patch_offset_y);
        }

		template <typename T, typename U> CUDA_CALLABLE
		void map(U* rx, U* ry, const T x, const T y, const T sx, const T sy) const noexcept {
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

		template <typename T>
		auto map(T local_x, T local_y, const Point2du& patchOffset, const Size32u& patchSize) const noexcept {
			FRAC_ASSERT(local_x >= 0 && local_x < patchSize.x());
			FRAC_ASSERT(local_y >= 0 && local_y < patchSize.y());
			const Point2d<T> p = this->map(local_x, local_y, patchSize);
			return Point2d<T>(p.x() + patchOffset.x(), p.y() + patchOffset.y());
		}
		template <typename T> CUDA_CALLABLE
		Point2d<T> map(const T x, const T y, const Size32u& s) const noexcept {
			static constexpr int __map_lookup[8][8] = {
				/*ID*/{ 1, 0, 0, 0,  0, 1, 0, 0 },
				/*90*/{ 0, 1, 0, 0,  -1, 0, 1, 0 },
				/*180*/{ -1, 0, 1, 0,  0, -1, 0, 1 },
				/*270*/{ 0, -1, 0, 1,  1, 0, 0, 0 },
				/*flip*/{ 1, 0, 0, 0,   0, -1, 0, 1 },
				/*fl 90*/{ 0, 1, 0, 0,   1, 0, 0, 0 },
				/*fl 180*/{ -1, 0, 1, 0,  0, 1, 0, 0 },
				/*fl 270*/{ 0, -1, 0, 1, -1, 0, 1, 0 }
			};
			return Point2d<T>{ __map_lookup[_type][0] * x + __map_lookup[_type][1] * y + __map_lookup[_type][2] * (s.x() - 1) + __map_lookup[_type][3] * (s.y() - 1), 
			__map_lookup[_type][4] * x + __map_lookup[_type][5] * y + __map_lookup[_type][6] * (s.x() - 1) + __map_lookup[_type][7] * (s.y() - 1)};
		}
		void copy(const Image& source, Image& target, const double contrast = 1.0, const double brightness = 0.0) const;
        void copy(const Frac2::ImagePlane& source, 
            Frac2::ImagePlane& target, 
            const Frac2::GridItemBase& sourcePatch,
            const Frac2::GridItemBase& targetPatch,
            const double contrast = 1.0, 
            const double brightness = 0.0) const;
	private:
		Image _resize_nn(const Image& source, const Size32u& targetSize) const;
		Image _resize_b(const Image& source, const Size32u& targetSize) const;
	private:
		Type _type = Id;
	};
}

#endif
