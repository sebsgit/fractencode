#include "image.h"
#include <iostream>
#ifdef FRAC_WITH_AVX
extern "C" {
#include "immintrin.h"
}
#endif

using namespace Frac;

double ImageStatistics::sum(const Image& a) noexcept {
    double result = a.cache().get(ImageData::KeySum);
    if (result == -1.0) {
        result = 0.0;
#ifdef FRAC_WITH_AVX
        if (a.width() % 8 == 0) {
            __m128i zero = _mm_set1_epi8(0);
            __m128i total = zero;
            for (uint32_t y=0 ; y<a.height() ; ++y) {
                uint32_t column = 0;
                const Image::Pixel* data = a.data()->get() + y * a.stride();
                while (column < a.width()) {
                    __m128i row = _mm_loadu_si128((const __m128i*)(data + column));
                    row = _mm_unpacklo_epi8(row, zero);
                    total = _mm_add_epi16(row, total);
                    column += 8;
                }
            }
            int tmpi[4];
            _mm_storeu_si128((__m128i*)tmpi, total);
            uint16_t* tmp = reinterpret_cast<uint16_t*>(tmpi);
            result = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        } else {
            a.map([&](Image::Pixel v) { result += v; });
        }
#else
        a.map([&](Image::Pixel v) { result += v; });
#endif
        a.cache().put(ImageData::KeySum, result);
    }
    return result;
}
double ImageStatistics::mean(const Image& image) noexcept {
    auto& cache = image.cache();
    double result = cache.get(ImageData::KeyMean);
    if (result == -1.0) {
        result = sum(image) / image.size().area();
        cache.put(ImageData::KeyMean, result);
    }
    return result;
}
double ImageStatistics::variance(const Image& image) noexcept {
    const double av = mean(image);
    double result = image.cache().get(ImageData::KeyVariance);
    if (result == -1.0) {
        double sum = 0.0;
#ifdef FRAC_WITH_AVX
        if (image.width() % 8 == 0) {
            __m128 total = _mm_set1_ps(0.0f);
            __m128 mean128 = _mm_set1_ps(av);
            for (uint32_t y=0 ; y<image.height() ; ++y) {
                uint32_t column = 0;
                const Image::Pixel* data = image.data()->get() + y * image.stride();
                while (column < image.width()) {
                    __m128i row = _mm_loadu_si128((const __m128i*)(data + column));
                    row = _mm_unpacklo_epi8(row, _mm_set1_epi8(0));
                    __m128i row_lo = _mm_unpacklo_epi16(row, _mm_set1_epi8(0));
                    __m128i row_hi = _mm_unpackhi_epi16(row, _mm_set1_epi8(0));
                    __m128 row_lo_f = _mm_cvtepi32_ps(row_lo);
                    __m128 row_hi_f = _mm_cvtepi32_ps(row_hi);
                    row_lo_f = _mm_sub_ps(row_lo_f, mean128);
                    row_lo_f = _mm_mul_ps(row_lo_f, row_lo_f);
                    total = _mm_add_ps(total, row_lo_f);
                    row_hi_f = _mm_sub_ps(row_hi_f, mean128);
                    row_hi_f = _mm_mul_ps(row_hi_f, row_hi_f);
                    total = _mm_add_ps(total, row_hi_f);
                    column += 8;
                }
            }
            float tmp[4];
            _mm_storeu_ps(tmp, total);
            sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        } else {
            image.map([&](Image::Pixel p) { sum += (p - av) * (p - av); });
        }
#else
        image.map([&](Image::Pixel p) { sum += (p - av) * (p - av); });
#endif
        result = sum / image.size().area();
        image.cache().put(ImageData::KeyVariance, result);
    }
    return result;
}
