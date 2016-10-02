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

void assert_sse_m256_epi16(const __m256i sse_value, 
	uint16_t x15, uint16_t x14, uint16_t x13, uint16_t x12,
	uint16_t x11, uint16_t x10, uint16_t x9, uint16_t x8,
	uint16_t x7, uint16_t x6, uint16_t x5, uint16_t x4,
	uint16_t x3, uint16_t x2, uint16_t x1, uint16_t x0)
{
	uint16_t tmp[16] = { 0 };
	_mm256_storeu_si256((__m256i*)tmp, sse_value);
	__ASSERT_EQ(tmp[0], x0);
	__ASSERT_EQ(tmp[1], x1);
	__ASSERT_EQ(tmp[2], x2);
	__ASSERT_EQ(tmp[3], x3);
	__ASSERT_EQ(tmp[4], x4);
	__ASSERT_EQ(tmp[5], x5);
	__ASSERT_EQ(tmp[6], x6);
	__ASSERT_EQ(tmp[7], x7);
	__ASSERT_EQ(tmp[8], x8);
	__ASSERT_EQ(tmp[9], x9);
	__ASSERT_EQ(tmp[10], x10);
	__ASSERT_EQ(tmp[11], x11);
	__ASSERT_EQ(tmp[12], x12);
	__ASSERT_EQ(tmp[13], x13);
	__ASSERT_EQ(tmp[14], x14);
	__ASSERT_EQ(tmp[15], x15);
}

#undef __ASSERT_EQ

#endif
