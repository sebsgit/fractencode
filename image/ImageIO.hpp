#pragma once

#include "image/Image2.hpp"
#include <string>

namespace Frac2 {
    class ImageIO {
    public:
        static std::array<ImagePlane, 3> rgb2yuv(const uint8_t* rgb, uint32_t width, uint32_t height, uint32_t stride);
		static void rgb2yuv(const uint8_t* rgb, uint32_t width, uint32_t height, uint32_t stride,
						uint8_t* yBuff, uint32_t yStride,
						uint8_t* uBuff, uint32_t uStride,
						uint8_t* vBuff, uint32_t vStride) noexcept;

        static void yuv2rgb(const uint8_t* yBuff, uint32_t ywidth, uint32_t yheight, uint32_t ystride,
            const uint8_t* uBuff, uint32_t ustride,
            const uint8_t* vBuff, uint32_t vstride,
            uint8_t* rgb, uint32_t rgbStride);

        static std::array<ImagePlane, 3> loadImage(const std::string& path);
        template <int planeCount>
        static void saveImage(const Image2<planeCount>& image, const std::string& path);
        static void saveImage(const ImagePlane& plane, const std::string& path);
    };
}