#pragma once

#include "../simd_base.hpp"
extern "C" {
#include <arm_neon.h>
}

namespace priv {
template <unsigned short l0, unsigned short l1, unsigned short h0, unsigned short h1>
float32x4_t neon_shuffle(float32x4_t a, float32x4_t b)
{
    float __attribute__((aligned(16))) data[] = { vgetq_lane_f32(a, l0), vgetq_lane_f32(a, l1), vgetq_lane_f32(b, h0), vgetq_lane_f32(b, h1) };
    return vld1q_f32(data);
}

//TODO: more optimized versions
template <>
inline float32x4_t neon_shuffle<0, 1, 2, 3>(float32x4_t a, float32x4_t b)
{
    return vcombine_f32(vget_low_f32(a), vget_high_f32(b));
}

} // namespace priv

template <>
class simd<float, 4> : public simd_common<simd<float, 4>> {
public:
    using type = float;
    static constexpr size_t value_count = 4;

    simd() noexcept
        : _d(vdupq_n_f32(0))
    {
    }
    explicit simd(float value) noexcept
        : _d(vdupq_n_f32(value))
    {
    }
    explicit simd(float32x4_t value) noexcept
        : _d(value)
    {
    }
    simd(float s0, float s1, float s2, float s3) noexcept
    {
        const float __attribute__((aligned(16))) data[] = { s0, s1, s2, s3 };
        this->_d = vld1q_f32(data);
    }
    explicit simd(const float* input) noexcept
        : _d(vld1q_f32(input))
    {
    }

    void store(float* output, cache_coherence cache_flags = cache_coherence::coherent) const noexcept
    {
        this->store_aligned(output, cache_flags);
    }

    void store_aligned(float* output, cache_coherence cache_flags = cache_coherence::coherent) const noexcept
    {
        if (cache_flags == cache_coherence::coherent) {
            vst1q_f32(output, this->_d);
        } else if (cache_flags == cache_coherence::non_temporal) {
            vst1q_f32(output, this->_d);
        }
    }

    simd operator+(const simd& other) const noexcept
    {
        return simd{ vaddq_f32(this->_d, other._d) };
    }
    simd& operator+=(const simd& other) noexcept
    {
        return *this = *this + other;
    }
    simd operator-(const simd& other) const noexcept
    {
        return simd{ vsubq_f32(this->_d, other._d) };
    }
    simd& operator-=(const simd& other) noexcept
    {
        return *this = *this - other;
    }
    simd operator*(const simd& other) const noexcept
    {
        return simd{ vmulq_f32(this->_d, other._d) };
    }
    simd& operator*=(const simd& other) noexcept
    {
        return *this = *this * other;
    }
    simd operator/(const simd& other) const noexcept
    {
        float32x4_t recp = vrecpeq_f32(other._d);
        recp = vmulq_f32(vrecpsq_f32(other._d, recp), recp);
        recp = vmulq_f32(vrecpsq_f32(other._d, recp), recp);
        // a / b = a * (1/b)
        return simd{ vmulq_f32(this->_d, recp) };
    }
    simd& operator/=(const simd& other) noexcept
    {
        return *this = *this / other;
    }

    simd min(const simd& other) const noexcept
    {
        return simd{ vminq_f32(this->_d, other._d) };
    }
    simd max(const simd& other) const noexcept
    {
        return simd{ vmaxq_f32(this->_d, other._d) };
    }
    simd sqrt() const noexcept
    {
        auto xi = vrsqrteq_f32(this->_d);
        xi = vrsqrtsq_f32(this->_d * xi, xi) * xi;
        return simd{ vrsqrtsq_f32(this->_d * xi, xi) * xi * this->_d };
    }
    simd abs() const noexcept
    {
        return simd{ vabsq_f32(this->_d) };
    }

    uint32x4_t compare_native(const simd& other, compare_flags flag) const noexcept
    {
        switch (flag) {
        case compare_flags::equal:
            return vceqq_f32(this->_d, other._d);
        case compare_flags::lower:
            return vcltq_f32(this->_d, other._d);
        case compare_flags::lower_equal:
            return vcleq_f32(this->_d, other._d);
        case compare_flags::greater:
            return vcgtq_f32(this->_d, other._d);
        case compare_flags::greater_equal:
            return vcgeq_f32(this->_d, other._d);
        case compare_flags::not_equal:
            return vmvnq_u32(vceqq_f32(this->_d, other._d));
        }
        return uint32x4_t{};
    }

    simd compare(const simd& other, compare_flags flag) const noexcept
    {
        return simd{ vcvtq_f32_u32(this->compare_native(other, flag)) };
    }

    static simd horizontal_add(const simd& a, const simd& b) noexcept
    {
        return simd{ vcombine_f32(
            vpadd_f32(vget_low_f32(a._d), vget_high_f32(a._d)),
            vpadd_f32(vget_low_f32(b._d), vget_high_f32(b._d))) };
    }

    template <unsigned short l0, unsigned short l1, unsigned short h0, unsigned short h1>
    static simd shuffle(const simd& a, const simd& b) noexcept
    {
        return simd{ priv::neon_shuffle<l0, l1, h0, h1>(a._d, b._d) };
    }

    static simd unpack_low(const simd& a, const simd& b) noexcept
    {
        auto v = vzip_f32(vget_low_f32(a._d), vget_low_f32(b._d));
        return simd{ vcombine_f32(v.val[0], v.val[1]) };
    }
    static simd unpack_high(const simd& a, const simd& b) noexcept
    {
        auto v = vzip_f32(vget_high_f32(a._d), vget_high_f32(b._d));
        return simd{ vcombine_f32(v.val[0], v.val[1]) };
    }

    static void transpose(simd& r0, simd& r1, simd& r2, simd& r3) noexcept
    {
        float32x4x2_t t0 = vtrnq_f32(r0._d, r1._d);
        float32x4x2_t t1 = vtrnq_f32(r2._d, r3._d);
        r0._d = vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0]));
        r1._d = vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1]));
        r2._d = vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0]));
        r3._d = vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1]));
    }

private:
    float32x4_t _d;
};

using simdf4 = simd<float, 4>;
