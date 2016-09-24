#include "gaussian5x5.h"
#include "sampler.h"
#include <immintrin.h>
#include <iostream>

using namespace Frac;

static constexpr const float _kernel[] = {
	1.f,  4.f,  7.f,  4.f, 1.f,
	4.f, 16.f, 26.f, 16.f, 4.f,
	7.f, 26.f, 41.f, 26.f, 7.f,
	4.f, 16.f, 26.f, 16.f, 4.f,
	1.f,  4.f,  7.f,  4.f, 1.f,
#ifdef FRAC_WITH_AVX
	0.f, 0.f, 0.f
#endif
};

static constexpr const int _normalizationFactor = 273;

Image GaussianBlur5x5::process(const Image &image) const {
	AbstractBufferPtr<Image::Pixel> data = Buffer<Image::Pixel>::alloc(image.height() * image.stride());
	Image result(data, image.width(), image.height(), image.stride());
	auto resultPtr = data->get();
	const int w = image.width();
	const int h = image.height();
	const SamplerLinear sampler(image);
#ifdef FRAC_WITH_AVX
	__m256 kernel0 = _mm256_loadu_ps(_kernel);
	__m256 kernel1 = _mm256_loadu_ps(_kernel + 5);
	__m256 kernel2 = _mm256_loadu_ps(_kernel + 10);
	__m256 kernel3 = _mm256_loadu_ps(_kernel + 15);
	__m256 kernel4 = _mm256_loadu_ps(_kernel + 20);
	__m256i zero8 = _mm256_set1_epi8(0);
	for (int y=0 ; y<h ; ++y) {
		for (int x=0 ; x<w ; ++x) {
			float sum = 0.0f;
			if (x<2 || y<2 || x>=w-2 || y>=h-2) {
				for (int k=-2; k<=2 ; ++k) {
					for (int j=-2; j<=2 ; ++j) {
						const float p = (x+k<0 || x+k>=w || y+j<0 || y+j>=h) ? 0.0f : (1.0f*_kernel[k+2+(j+2)*5]*sampler(x+k, (y+j)));
						sum += p;
					}
				}
			} else {
				__m256i row0 = _mm256_loadu_si256((const __m256i*)(image.data()->get() + x - 2 + (y - 2) * image.stride()));
				__m256i row1 = _mm256_loadu_si256((const __m256i*)(image.data()->get() + x - 2 + (y - 1) * image.stride()));
				__m256i row2 = _mm256_loadu_si256((const __m256i*)(image.data()->get() + x - 2 + (y) * image.stride()));
				__m256i row3 = _mm256_loadu_si256((const __m256i*)(image.data()->get() + x - 2 + (y + 1) * image.stride()));
				__m256i row4 = _mm256_loadu_si256((const __m256i*)(image.data()->get() + x - 2 + (y + 2) * image.stride()));
				__m256i row0_16 = _mm256_unpacklo_epi8(row0, zero8);
				__m256i row1_16 = _mm256_unpacklo_epi8(row1, zero8);
				__m256i row2_16 = _mm256_unpacklo_epi8(row2, zero8);
				__m256i row3_16 = _mm256_unpacklo_epi8(row3, zero8);
				__m256i row4_16 = _mm256_unpacklo_epi8(row4, zero8);
				__m256i row0_32 = _mm256_unpacklo_epi16(row0_16, zero8);
				__m256i row1_32 = _mm256_unpacklo_epi16(row1_16, zero8);
				__m256i row2_32 = _mm256_unpacklo_epi16(row2_16, zero8);
				__m256i row3_32 = _mm256_unpacklo_epi16(row3_16, zero8);
				__m256i row4_32 = _mm256_unpacklo_epi16(row4_16, zero8);
				__m256 row0f = _mm256_cvtepi32_ps(row0_32);
				__m256 row1f = _mm256_cvtepi32_ps(row1_32);
				__m256 row2f = _mm256_cvtepi32_ps(row2_32);
				__m256 row3f = _mm256_cvtepi32_ps(row3_32);
				__m256 row4f = _mm256_cvtepi32_ps(row4_32);
				row0f = _mm256_mul_ps(row0f, kernel0);
				row1f = _mm256_mul_ps(row1f, kernel1);
				row2f = _mm256_mul_ps(row2f, kernel2);
				row3f = _mm256_mul_ps(row3f, kernel3);
				row4f = _mm256_mul_ps(row4f, kernel4);
				row0f = _mm256_add_ps(row0f, row0f);
				row0f = _mm256_add_ps(row0f, row1f);
				row0f = _mm256_add_ps(row0f, row2f);
				row0f = _mm256_add_ps(row0f, row3f);
				row0f = _mm256_add_ps(row0f, row4f);
				float tmp[8];
				_mm256_storeu_ps(tmp, row0f);
				sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
			}
			resultPtr[x + y*image.stride()] = convert<Image::Pixel>(sum / _normalizationFactor);
		}
	}
#else
	for (int y=0 ; y<h ; ++y) {
		for (int x=0 ; x<w ; ++x) {
			float sum = 0.0f;
			for (int k=-2; k<=2 ; ++k) {
				for (int j=-2; j<=2 ; ++j) {
					const float p = (x+k<0 || x+k>=w || y+j<0 || y+j>=h) ? 0.0 : (1.0f*_kernel[k+2+(j+2)*5]*sampler(x+k, (y+j)));
					sum += p;
				}
			}
			resultPtr[x + y*image.stride()] = convert<Image::Pixel>(sum / _normalizationFactor);
		}
	}
#endif
	return result;
}
