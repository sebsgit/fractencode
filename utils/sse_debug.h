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

#define frac_m256_interleave2_epi16(hi, lo) _mm256_set_epi16(hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo)
#define frac_m256_interleave4_epi16(hi1, hi0, lo1, lo0) _mm256_set_epi16(hi1, hi0, lo1, lo0, hi1, hi0, lo1, lo0, hi1, hi0, lo1, lo0, hi1, hi0, lo1, lo0)

extern void assert_sse_m128_epi16(const __m128i sse_value, uint16_t x7, uint16_t x6, uint16_t x5, uint16_t x4,
	uint16_t x3, uint16_t x2, uint16_t x1, uint16_t x0);

extern void assert_sse_m256_epi16(const __m256i sse_value,
	uint16_t x15, uint16_t x14, uint16_t x13, uint16_t x12,
	uint16_t x11, uint16_t x10, uint16_t x9, uint16_t x8,
	uint16_t x7, uint16_t x6, uint16_t x5, uint16_t x4,
	uint16_t x3, uint16_t x2, uint16_t x1, uint16_t x0);

#endif

#endif
