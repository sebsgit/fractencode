#include "sse_debug.h"

#ifdef FRAC_WITH_AVX

#include <iostream>

#define __ASSERT_EQ(x, y) if ((x) != (y)) {	\
	std::cout << __FUNCTION__ << ' ' << (x) << " != " << (y) << '\n'; }

void assert_sse_m128_epi16(const __m128i sse_value, uint16_t x7, uint16_t x6, uint16_t x5, uint16_t x4,
	uint16_t x3, uint16_t x2, uint16_t x1, uint16_t x0)
{
	uint16_t tmp[8] = { 0 };
	_mm_storeu_si128((__m128i*)tmp, sse_value);
	__ASSERT_EQ(tmp[0], x0);
	__ASSERT_EQ(tmp[1], x1);
	__ASSERT_EQ(tmp[2], x2);
	__ASSERT_EQ(tmp[3], x3);
	__ASSERT_EQ(tmp[4], x4);
	__ASSERT_EQ(tmp[5], x5);
	__ASSERT_EQ(tmp[6], x6);
	__ASSERT_EQ(tmp[7], x7);
}

#endif
