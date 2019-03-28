#pragma once

#include "../simd_base.hpp"

template <>
class simd<float, 4> : public simd_common<simd<float, 4>> {
public:
    using type = float;
    static constexpr size_t value_count = 4;

    simd() noexcept
        : _d(_mm_setzero_ps())
    {
    }
    explicit simd(float value) noexcept
        : _d(_mm_set1_ps(value))
    {
    }
    simd(float s0, float s1, float s2, float s3) noexcept
        : _d(_mm_setr_ps(s0, s1, s2, s3))
    {
    }
    explicit simd(__m128 value) noexcept
        : _d(value)
    {
    }
    explicit simd(const float* input) noexcept
        : _d(is_aligned(input, 16) ? _mm_load_ps(input) : _mm_loadu_ps(input))
    {
    }

    void store(float* output, cache_coherence cache_flags = cache_coherence::coherent) const noexcept
    {
        if (cache_flags == cache_coherence::coherent) {
            if (is_aligned(output, 16)) {
                _mm_store_ps(output, this->_d);
            } else {
                _mm_storeu_ps(output, this->_d);
            }
        } else if (cache_flags == cache_coherence::non_temporal) {
            //TODO: assert aligned
            _mm_stream_ps(output, this->_d);
        }
    }

    void store_aligned(float* output, cache_coherence cache_flags = cache_coherence::coherent) const noexcept
    {
        if (cache_flags == cache_coherence::coherent) {
            _mm_storeu_ps(output, this->_d);
        } else if (cache_flags == cache_coherence::non_temporal) {
            _mm_stream_ps(output, this->_d);
        }
    }

    simd operator+(const simd& other) const noexcept
    {
        return simd{ _mm_add_ps(this->_d, other._d) };
    }
    simd& operator+=(const simd& other) noexcept
    {
        return *this = *this + other;
    }
    simd operator-(const simd& other) const noexcept
    {
        return simd{ _mm_sub_ps(this->_d, other._d) };
    }
    simd& operator-=(const simd& other) noexcept
    {
        return *this = *this - other;
    }
    simd operator*(const simd& other) const noexcept
    {
        return simd{ _mm_mul_ps(this->_d, other._d) };
    }
    simd& operator*=(const simd& other) noexcept
    {
        return *this = *this * other;
    }
    simd operator/(const simd& other) const noexcept
    {
        return simd{ _mm_div_ps(this->_d, other._d) };
    }
    simd& operator/=(const simd& other) noexcept
    {
        return *this = *this / other;
    }

    simd min(const simd& other) const noexcept
    {
        return simd{ _mm_min_ps(this->_d, other._d) };
    }
    simd max(const simd& other) const noexcept
    {
        return simd{ _mm_max_ps(this->_d, other._d) };
    }
    simd sqrt() const noexcept
    {
        return simd{ _mm_sqrt_ps(this->_d) };
    }
    simd abs() const noexcept
    {
        const auto mask = _mm_set1_ps(-1 * 0.0f);
        return simd{ _mm_andnot_ps(mask, this->_d) };
    }

    simd compare(const simd& other, compare_flags flag) const noexcept
    {
        switch (flag) {
        case compare_flags::equal:
            return simd{ _mm_cmpeq_ps(this->_d, other._d) };
        case compare_flags::lower:
            return simd{ _mm_cmplt_ps(this->_d, other._d) };
        case compare_flags::lower_equal:
            return simd{ _mm_cmple_ps(this->_d, other._d) };
        case compare_flags::greater:
            return simd{ _mm_cmpgt_ps(this->_d, other._d) };
        case compare_flags::greater_equal:
            return simd{ _mm_cmpge_ps(this->_d, other._d) };
        case compare_flags::not_equal:
            return simd{ _mm_cmpneq_ps(this->_d, other._d) };
        }
        return simd{};
    }

    static simd horizontal_add(const simd& a, const simd& b) noexcept
    {
        //SSE3
        return simd{ _mm_hadd_ps(a._d, b._d) };
    }

    template <unsigned short l0, unsigned short l1, unsigned short h0, unsigned short h1>
    static simd shuffle(const simd& a, const simd& b) noexcept
    {
        return simd{ _mm_shuffle_ps(a._d, b._d, _MM_SHUFFLE(h1, h0, l1, l0)) };
    }
    static simd unpack_low(const simd& a, const simd& b) noexcept
    {
        return simd{ _mm_unpacklo_ps(a._d, b._d) };
    }
    static simd unpack_high(const simd& a, const simd& b) noexcept
    {
        return simd{ _mm_unpackhi_ps(a._d, b._d) };
    }
    static void transpose(simd& r0, simd& r1, simd& r2, simd& r3) noexcept
    {
        _MM_TRANSPOSE4_PS(r0._d, r1._d, r2._d, r3._d);
    }

private:
    __m128 _d;
};

using simdf4 = simd<float, 4>;
