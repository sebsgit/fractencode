#include "process/sobel.h"
#ifdef FRAC_WITH_AVX
#include <immintrin.h>
#endif
#include <cmath>

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
static const int16_t kernel_x_simd[] __attribute__ ((aligned (16))) = { -1, 1, -2, 2, -1, 1, 0, 0 };
static const int16_t kernel_y_simd[] __attribute__ ((aligned (16))) = { -1, -2, -1, 1, 2, 1, 0, 0 };
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
                const uint8_t a = image.data()->get()[(x - 1) + (y - 1) * image.stride()];
                const uint8_t c = image.data()->get()[(x + 1) + (y - 1) * image.stride()];
                const uint8_t d = image.data()->get()[(x - 1) + (y) * image.stride()];
                const uint8_t e = image.data()->get()[(x + 1) + (y) * image.stride()];
                const uint8_t f = image.data()->get()[(x - 1) + (y + 1) * image.stride()];
                const uint8_t h = image.data()->get()[(x + 1) + (y + 1) * image.stride()];
                __m128i data_x = _mm_set_epi16(0, 0, h, f, e, d, c, a);
                data_x = _mm_mullo_epi16(data_x, kernel_x128i);
                __m128i row0 = _mm_loadu_si128((const __m128i*)(image.data()->get() + x - 1 + (y - 1) * image.stride()));
                __m128i row2 = _mm_loadu_si128((const __m128i*)(image.data()->get() + x - 1 + (y + 1) * image.stride()));
                row0 = _mm_unpacklo_epi8(row0, _mm_setzero_si128());
                row2 = _mm_unpacklo_epi8(row2, _mm_setzero_si128());
                row2 = _mm_bslli_si128(row2, 6);
                row0 = _mm_mullo_epi16(row0, _mm_set_epi16(0, 0, 0, 0, 0, 1, 1, 1));
                __m128i data_y = _mm_mullo_epi16(_mm_add_epi16(row0, row2), kernel_y128i);
                int16_t tmp[8];
                _mm_storeu_si128((__m128i*)tmp, data_x);
                derivative.dx = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5];
                _mm_storeu_si128((__m128i*)tmp, data_y);
                derivative.dy = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5];
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
