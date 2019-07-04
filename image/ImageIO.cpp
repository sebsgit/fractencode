#include "image/ImageIO.hpp"

#include "thirdparty/stb_image/stb_image.h"
#include "thirdparty/stb_image/stb_image_write.h"

#include <gsl/gsl_util>

using namespace Frac2;

static uint8_t clamp(const double x) noexcept {
    return x < 0.0 ? 0 : x > 255 ? 255 : (uint8_t)x;
}

static uint32_t align64(uint32_t v) noexcept {
    return (v % 64 == 0) ? v : (v + 64 - v % 64);
}

std::array<ImagePlane, 3> ImageIO::rgb2yuv(gsl::not_null<const uint8_t*> rgb, uint32_t width, uint32_t height, uint32_t stride)
{
    const uint32_t padding = 32;
	const uint32_t yStride = align64(width) + padding;
    const uint32_t uvStride = align64(width / 2) + padding;

    std::array<ImagePlane, 3> result {
		ImagePlane({width, height}, yStride),
        ImagePlane({width / 2, height / 2}, uvStride),
		ImagePlane({width / 2, height / 2}, uvStride)
	};

	rgb2yuv({rgb, stride * height * 3}, width, height, stride,
		{result[0].data(), yStride * height}, result[0].stride(),
		{result[1].data(), uvStride * height / 2}, result[1].stride(),
		{result[2].data(), uvStride * height / 2}, result[2].stride()
	);

    return result;
}

void ImageIO::rgb2yuv(gsl::span<const uint8_t> rgb, uint32_t width, uint32_t height, uint32_t stride,
						gsl::span<uint8_t> yBuff, uint32_t yStride,
						gsl::span<uint8_t> uBuff, uint32_t uStride,
						gsl::span<uint8_t> vBuff, uint32_t vStride) noexcept
{
	for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            auto r = rgb[x * 3 + y * stride + 0];
            auto g = rgb[x * 3 + y * stride + 1];
            auto b = rgb[x * 3 + y * stride + 2];
            auto yp = 0.299 * r + 0.587 * g + 0.114 * b;
            auto up = -0.169 * r - 0.331 * g + 0.499 * b + 128;
            auto vp = 0.499 * r - 0.418 * g - 0.0813 * b + 128;
            yBuff[x + y * yStride] = clamp(yp);
            uBuff[(x / 2) + ((y / 2)  * uStride)] = clamp(up);
            vBuff[(x / 2) + ((y / 2) * vStride)] = clamp(vp);
        }
    }
}

std::array<ImagePlane, 3> ImageIO::loadImage(const std::string& path)
{
    int components = 0;
    int w, h;
    auto data = std::unique_ptr<uint8_t, decltype(&::free)>(stbi_load(path.c_str(), &w, &h, &components, 3), &free);
    return rgb2yuv(data.get(), w, h, w * 3);
}

void ImageIO::yuv2rgb(const uint8_t* yBuff, uint32_t ywidth, uint32_t yheight, uint32_t ystride,
    const uint8_t* uBuff, uint32_t ustride,
    const uint8_t* vBuff, uint32_t vstride,
    uint8_t* rgb, uint32_t rgbStride)
{
    for (size_t y = 0; y < yheight; ++y) {
        for (size_t x = 0; x < ywidth; ++x) {
            gsl::span<unsigned char, 3> ptr = {rgb + (x * 3 + y * rgbStride * 3), 3};
            double yp = yBuff[x + y * ystride];
            double up = uBuff[(x / 2) + (y / 2) * ustride];
            double vp = vBuff[(x / 2) + (y / 2) * vstride];
            ptr[0] = clamp(yp + 1.402 * (vp - 128));
            ptr[1] = clamp(yp - 0.344 * (up - 128) - 0.714 * (vp - 128));
            ptr[2] = clamp(yp + 1.772 * (up - 128));
        }
    }
}

template <>
void ImageIO::saveImage<3>(const Image2<3>& image, const std::string& path)
{
    const auto& yPlane = image.plane(0);
    const uint32_t rgbStride = yPlane.stride();
    std::vector<uint8_t> rgbData(yPlane.size().y() * rgbStride * 3);
    yuv2rgb(yPlane.data(), yPlane.size().x(), yPlane.size().y(), yPlane.stride(),
        image.plane(1).data(), image.plane(1).stride(),
        image.plane(2).data(), image.plane(2).stride(),
        rgbData.data(), rgbStride);
    stbi_write_png(path.c_str(), yPlane.size().x(), yPlane.size().y(), 3, rgbData.data(), rgbStride * 3);
}

void ImageIO::saveImage(const ImagePlane& image, const std::string& path)
{
    stbi_write_png(path.c_str(), image.size().x(), image.size().y(), 1, image.data(), image.stride());
}
