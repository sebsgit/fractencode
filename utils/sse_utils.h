#pragma once

#ifdef FRAC_WITH_AVX

#include "sse_debug.h"

#define frac_m256_interleave2_epi16(hi, lo) _mm256_set_epi16(hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo)
#define frac_m256_interleave4_epi16(hi1, hi0, lo1, lo0) _mm256_set_epi16(hi1, hi0, lo1, lo0, hi1, hi0, lo1, lo0, hi1, hi0, lo1, lo0, hi1, hi0, lo1, lo0)

extern float frac_hsum256_ps(__m256 value);

#endif
