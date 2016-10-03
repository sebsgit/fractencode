#ifndef FRAC_SSE_DEBUG
#define FRAC_SSE_DEBUG

#ifdef FRAC_WITH_AVX

#include <immintrin.h>
#include <inttypes.h>

#ifdef _MSC_VER
#define ALIGN_SPEC __declspec(align(16))
#define ALIGN_ATTR
#define byteshift_left _mm_slli_si128
#else
#define ALIGN_SPEC
#define ALIGN_ATTR __attribute__ ((aligned (16)))
#define byteshift_left _mm_bslli_si128
#endif
#define FRAC_ALIGNED_16(what) ALIGN_SPEC what ALIGN_ATTR

extern void assert_sse_m128_epi16(const __m128i sse_value, uint16_t x7, uint16_t x6, uint16_t x5, uint16_t x4,
	uint16_t x3, uint16_t x2, uint16_t x1, uint16_t x0);

extern void assert_sse_m256_epi16(const __m256i sse_value,
	uint16_t x15, uint16_t x14, uint16_t x13, uint16_t x12,
	uint16_t x11, uint16_t x10, uint16_t x9, uint16_t x8,
	uint16_t x7, uint16_t x6, uint16_t x5, uint16_t x4,
	uint16_t x3, uint16_t x2, uint16_t x1, uint16_t x0);

extern void assert_sse_m256_epi16(const __m256i sse_value, const uint8_t* imageRow);
extern void assert_sse_m256_epi16(const __m256i sse_value, const uint16_t* data);

extern void assert_sse_m256_epi16_sum(const __m256i sse_value, const uint8_t* row0, const uint8_t* row1, const uint8_t* row2, const uint8_t* row3,
	const uint8_t* row4, const uint8_t* row5, const uint8_t* row6, const uint8_t* row7);

template <typename T>
void assert_eq(const T a, const T b) {
	if (a != b) {
		std::cout << a << ' ' << b << '\n';
		exit(0);
	}
}

#endif

#endif
