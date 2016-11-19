#include "process/sobel.h"
#include "Config.h"
#ifdef FRAC_WITH_AVX
#include <immintrin.h>
#include "sse_debug.h"
#endif
#include <cmath>
#include <inttypes.h>

using namespace Frac;

static const int _kernel_x[] = {
	-1, 0, 1,
	-2, 0, 2,
	-1, 0, 1
};

static const int _kernel_y[] = {
	-1, -2, -1,
	0, 0, 0,
	1, 2, 1
};

#ifdef FRAC_WITH_AVX
ALIGN_SPEC static const int16_t kernel_y_simd[] ALIGN_ATTR = { -1, -2, -1, 1, 2, 1, 0, 0 };
ALIGN_SPEC static const int16_t kernel_x_simd[] ALIGN_ATTR = { -1, -2, 1, 2, -1, 0, 1, 0 };
#endif

Image SobelOperator::process(const Image &image) const {
	const auto buffer = this->calculate(image);
	auto imageData = Buffer<Image::Pixel>::alloc(image.width() * image.height());
	Image result(imageData, image.width(), image.height(), image.width());
	for (uint32_t y=0 ; y<image.height() ; ++y) {
		for (uint32_t x=0 ; x<image.width() ; ++x) {
			const result_t dv = buffer->get()[x + y * image.width()];
			const uint8_t gradient = (uint8_t)sqrt( dv.dx*dv.dx + dv.dy*dv.dy );
			imageData->get()[x + y * image.width()] = gradient;
		}
	}
	return result;
}

AbstractBufferPtr<SobelOperator::result_t> SobelOperator::calculate(const Image &image) const {
	auto result = Buffer<result_t>::alloc(image.width() * image.height());
#ifdef FRAC_WITH_AVX
   __m128i kernel_x128i = _mm_load_si128((const __m128i*)kernel_x_simd);
   __m128i kernel_y128i = _mm_load_si128((const __m128i*)kernel_y_simd);
#endif
	for (int y=0 ; y<(int)image.height() ; ++y) {
		for (int x=0 ; x<(int)image.width() ; ++x) {
			result_t derivative;
#ifdef FRAC_WITH_AVX
			if (y==0 || x==0 || y==(int)image.height()-1 || x==(int)image.width()-1) {
				for (int i=-1 ; i<=1 ; ++i) {
					for (int j=-1 ; j<=1 ; ++j) {
						const int opx = _kernel_x[(j + 1) + (i + 1)*3];
						const int opy = _kernel_y[(j + 1) + (i + 1)*3];
						const int xs = x + j < 0 ? 0 : x + j >= (int)image.width() ? image.width() - 1 : x + j;
						const int ys = y + i < 0 ? 0 : y + i >= (int)image.height() ? image.height() - 1 : y + i;
						const auto value = image.data()->get()[xs + ys * image.stride()];
						derivative.dx += (int)value * opx;
						derivative.dy += (int)value * opy;
					}
				}
			} else {
				__m128i row0 = _mm_loadu_si128((const __m128i*)(image.data()->get() + x - 1 + (y - 1) * image.stride()));
				__m128i row1 = _mm_loadu_si128((const __m128i*)(image.data()->get() + x - 1 + (y) * image.stride()));
				__m128i row2 = _mm_loadu_si128((const __m128i*)(image.data()->get() + x - 1 + (y + 1) * image.stride()));

				row0 = _mm_unpacklo_epi8(row0, _mm_setzero_si128());
				row1 = _mm_unpacklo_epi8(row1, _mm_setzero_si128());
				row2 = _mm_unpacklo_epi8(row2, _mm_setzero_si128());

				__m128i mask_x = _mm_set_epi16(0, 0, 0, 0, 0, 1, 0, 1);
				__m128i row0_x = _mm_mullo_epi16(row0, mask_x);
				__m128i row1_x = _mm_mullo_epi16(row1, mask_x);
				__m128i row2_x = _mm_mullo_epi16(row2, mask_x);

				row1_x = byteshift_left(row1_x, 2);
				row2_x = byteshift_left(row2_x, 8);
				row0_x = _mm_add_epi16(row0_x, row1_x);
				row0_x = _mm_add_epi16(row0_x, row2_x);
				__m128i data_x = _mm_mullo_epi16(row0_x, kernel_x128i);

				row2 = byteshift_left(row2, 6);
				row0 = _mm_mullo_epi16(row0, _mm_set_epi16(0, 0, 0, 0, 0, 1, 1, 1));
				__m128i data_y = _mm_mullo_epi16(_mm_add_epi16(row0, row2), kernel_y128i);

				FRAC_ALIGNED_16(int16_t tmp[8]);
				_mm_store_si128((__m128i*)tmp, data_x);
				derivative.dx = (float)tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6];
				_mm_store_si128((__m128i*)tmp, data_y);
				derivative.dy = (float)tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5];
			}
#else
			for (int i=-1 ; i<=1 ; ++i) {
				for (int j=-1 ; j<=1 ; ++j) {
					const int opx = _kernel_x[(j + 1) + (i + 1)*3];
					const int opy = _kernel_y[(j + 1) + (i + 1)*3];
					const int xs = x + j < 0 ? 0 : x + j >= (int)image.width() ? image.width() - 1 : x + j;
					const int ys = y + i < 0 ? 0 : y + i >= (int)image.height() ? image.height() - 1 : y + i;
					const auto value = image.data()->get()[xs + ys * image.stride()];
					derivative.dx += (int)value * opx;
					derivative.dy += (int)value * opy;
				}
			}
#endif
			result->get()[x + y * image.width()] = derivative;
		}
	}
	return result;
}


Image NonMaximumSuppressionOperator::edgeImage(AbstractBufferPtr<SobelOperator::result_t> gradients, uint32_t imageWidth, uint32_t imageHeight) const {
	uint32_t resultStride = imageWidth + 64;
	AbstractBufferPtr<Image::Pixel> resultBuffer = Buffer<Image::Pixel>::alloc(resultStride * imageHeight);
	resultBuffer->memset(0);
	Image result(resultBuffer, imageWidth, imageHeight, resultStride);
	auto oldCallback = this->_pixelCallback;
	this->_pixelCallback = [&](uint32_t x, uint32_t y, bool isStrongEdge) {
		resultBuffer->get()[x + y * resultStride] = isStrongEdge ? 255 : 100;
	};
	this->process(gradients, imageWidth, imageHeight);
	this->_pixelCallback = oldCallback;
	return result;
}

void NonMaximumSuppressionOperator::process(AbstractBufferPtr<SobelOperator::result_t> gradients, uint32_t imageWidth, uint32_t imageHeight) const {
	static const auto halfQuarter = 22.5f;
	for (uint32_t y=1 ; y<imageHeight - 1 ; ++y) {
		for (uint32_t x=1 ; x<imageWidth - 1 ; ++x) {
			const auto g = gradients->get()[x + y * imageWidth];
			const auto magnitude_xy = sqrt(g.dx * g.dx + g.dy * g.dy);
			if (magnitude_xy > this->_loCutoff) {
				const auto angle_xy = ((atan2(g.dy, g.dx) * 180) / 3.14) + 180;
				uint32_t x01 = x, x11 = x, y01 = y, y11 = y;
				if (abs(angle_xy - 0) < halfQuarter || abs(angle_xy - 180) < halfQuarter || abs(angle_xy - 360) < halfQuarter) {
					y01++;
					y11--;
				} else if ((abs(angle_xy - 45) < halfQuarter) || abs(angle_xy - 225) < halfQuarter) {
					x01--;
					y01++;
					x11++;
					y11--;
				} else if ((abs(angle_xy - 90) < halfQuarter) || abs(angle_xy - 270) < halfQuarter) {
				   x01++;
				   x11--;
				} else {
				   x01++;
				   y01++;
				   x11--;
				   y11--;
				}
				const auto g01 = gradients->get()[x01 + y01 * imageWidth];
				const auto g11 = gradients->get()[x11 + y11 * imageWidth];
				const auto magnitude_01 = sqrt(g01.dx*g01.dx + g01.dy*g01.dy);
				const auto magnitude_11 = sqrt(g11.dx*g11.dx + g11.dy*g11.dy);
				if (magnitude_xy > magnitude_01 && magnitude_xy > magnitude_11)
					this->_pixelCallback(x, y, magnitude_xy > _hiCutoff);
			}
		}
	}
}
