#ifndef FRAC_TRANSFORM_H_
#define FRAC_TRANSFORM_H_

#include "utils/size.hpp"
#include "utils/point2d.hpp"
#include "gpu/cuda/CudaConf.h"
#include "utils/Assert.hpp"
#include <iostream>
#include <array>

namespace Frac2 {
    class GridItemBase;
};

namespace Frac {
    enum class TransformType {
        Id = 0,
        Rotate_90,
        Rotate_180,
        Rotate_270,
        Flip,
        Flip_Rotate_90,
        Flip_Rotate_180,
        Flip_Rotate_270
    };

    enum class Interpolation {
        NearestNeighbor,
        Bilinear
    };

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

    template <TransformType type>
	class Transform {
        static constexpr int _type = static_cast<int>(type);
	public:
        constexpr Size32u map(const Size32u& s) const noexcept {
            switch(type) {
            case TransformType::Rotate_90:
            case TransformType::Rotate_270:
            case TransformType::Flip_Rotate_90:
            case TransformType::Flip_Rotate_270:
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
		std::array<std::ptrdiff_t, 4> generateSampleOffsets(uint32_t imageStride, uint32_t local_x, uint32_t local_y, const Point2du& patchOffset, const Size32u& patchSize) const noexcept
		{
			FRAC_ASSERT(local_x >= 0 && local_x + 1 < patchSize.x());
			FRAC_ASSERT(local_y >= 0 && local_y + 1 < patchSize.y());
			const auto patch_offset_x = patchOffset.x() + __map_lookup[_type][0] * local_x + __map_lookup[_type][1] * local_y + __map_lookup[_type][2] * (patchSize.x() - 1) + __map_lookup[_type][3] * (patchSize.y() - 1);
			const auto patch_stride = imageStride * (patchOffset.y() + __map_lookup[_type][4] * local_x + __map_lookup[_type][5] * local_y + __map_lookup[_type][6] * (patchSize.x() - 1) + __map_lookup[_type][7] * (patchSize.y() - 1));

			const std::ptrdiff_t offset_p0 = patch_stride + patch_offset_x;
			const std::ptrdiff_t offset_p1 = (__map_lookup[_type][4]) * imageStride + patch_stride + __map_lookup[_type][0] + patch_offset_x;
			const std::ptrdiff_t offset_p2 = (__map_lookup[_type][5]) * imageStride + patch_stride + __map_lookup[_type][1] + patch_offset_x;
			const std::ptrdiff_t offset_p3 = (__map_lookup[_type][4] + __map_lookup[_type][5]) * imageStride + patch_stride + __map_lookup[_type][0] + __map_lookup[_type][1] + patch_offset_x;

			return { offset_p0, offset_p1, offset_p2, offset_p3 };
		}

		template <typename T> CUDA_CALLABLE
		Point2d<T> map(const T x, const T y, const Size32u& s) const noexcept {
			return Point2d<T>{ __map_lookup[_type][0] * x + __map_lookup[_type][1] * y + __map_lookup[_type][2] * (s.x() - 1) + __map_lookup[_type][3] * (s.y() - 1), 
			__map_lookup[_type][4] * x + __map_lookup[_type][5] * y + __map_lookup[_type][6] * (s.x() - 1) + __map_lookup[_type][7] * (s.y() - 1)};
		}
	};
}

#endif
