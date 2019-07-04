#pragma once

#include "image/Image2.hpp"
#include <string>
#include <gsl/pointers>
#include <gsl/span>

namespace Frac2 {
    class ImageIO {
    public:
        static std::array<ImagePlane, 3> rgb2yuv(gsl::not_null<const uint8_t*> rgb, uint32_t width, uint32_t height, uint32_t stride);
		static void rgb2yuv(gsl::span<const uint8_t> rgb, uint32_t width, uint32_t height, uint32_t stride,
						gsl::span<uint8_t> yBuff, uint32_t yStride,
						gsl::span<uint8_t> uBuff, uint32_t uStride,
						gsl::span<uint8_t> vBuff, uint32_t vStride) noexcept;

        static void yuv2rgb(gsl::span<const uint8_t> yBuff, uint32_t ywidth, uint32_t yheight, uint32_t ystride,
            gsl::span<const uint8_t> uBuff, uint32_t ustride,
            gsl::span<const uint8_t> vBuff, uint32_t vstride,
            gsl::span<uint8_t> rgb, uint32_t rgbStride);

        static std::array<ImagePlane, 3> loadImage(const std::string& path);
        template <int planeCount>
        static void saveImage(const Image2<planeCount>& image, const std::string& path);
        static void saveImage(const ImagePlane& plane, const std::string& path);
    };
}
