#include "sse_utils.h"
#include "Config.h"

#ifdef FRAC_WITH_AVX

inline float frac_hsum256_ps(__m256 value) {
	__m128 value_ps = _mm_add_ps(_mm256_extractf128_ps(value, 0), _mm256_extractf128_ps(value, 1));
	value_ps = _mm_hadd_ps(value_ps, value_ps);
	value_ps = _mm_hadd_ps(value_ps, value_ps);
	return _mm_cvtss_f32(value_ps);
}

#endif
