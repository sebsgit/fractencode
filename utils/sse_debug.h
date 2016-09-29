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

extern void assert_sse_m128_epi16(const __m128i sse_value, uint16_t x7, uint16_t x6, uint16_t x5, uint16_t x4,
	uint16_t x3, uint16_t x2, uint16_t x1, uint16_t x0);

#endif

#endif
