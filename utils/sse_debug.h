#ifndef FRAC_SSE_DEBUG
#define FRAC_SSE_DEBUG

#ifdef FRAC_WITH_AVX

#include <immintrin.h>
#include <inttypes.h>

extern void assert_sse_m128_epi16(const __m128i sse_value, uint16_t x7, uint16_t x6, uint16_t x5, uint16_t x4,
	uint16_t x3, uint16_t x2, uint16_t x1, uint16_t x0);

#endif

#endif
