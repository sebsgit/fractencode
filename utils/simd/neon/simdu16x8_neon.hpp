#pragma once

#include "../simd_base.hpp"
extern "C" {
#include <arm_neon.h>
}

#include <iostream>

template <>
class simd<uint16_t, 8> : public simd_common<simd<uint16_t, 8>> {
public:
    using type = uint16_t;
    static constexpr size_t value_count = 8;

    simd() noexcept
        : _d(vdupq_n_u16(0))
    {
    }
    explicit simd(uint16_t value) noexcept
        : _d(vdupq_n_u16(value))
    {
    }
    simd(uint16_t s0, uint16_t s1, uint16_t s2, uint16_t s3,
        uint16_t s4, uint16_t s5, uint16_t s6, uint16_t s7) noexcept
    {
        alignas(16) const uint16_t data[] = { s0, s1, s2, s3, s4, s5, s6, s7 };
        this->_d = vld1q_u16(data);
    }
    explicit simd(uint16x8_t value) noexcept
        : _d(value)
    {
    }
    explicit simd(const uint16_t* input) noexcept
        : _d(vld1q_u16(input))
    {
    }

    explicit simd(const uint16_t* inputLo, const uint16_t* inputHi) noexcept
        : simd(inputLo[0], inputLo[1], inputLo[2], inputLo[3],
            inputHi[0], inputHi[1], inputHi[2], inputHi[3])
    {
    }
    /**
        Loads 8 unsigned 8 bit integers from memory and converts them to 16 bit values.
        @param input Memory to load the values from. This pointer needs to point to allocation of at least 8 bytes.
    */
    explicit simd(const uint8_t* input) noexcept
        : simd(input[0], input[1], input[2], input[3],
            input[4], input[5], input[6], input[7])
    {
    }

    void store(uint16_t* output) const noexcept
    {
        if (is_aligned(output, 16)) {
            this->store_aligned(output);
        }
        else {
            alignas(16) uint16_t tmp[8];
            this->store_aligned(tmp);
            memcpy(output, tmp, 8 * sizeof(tmp[0]));
        }
    }

    void store_aligned(uint16_t* output) const noexcept
    {
        vst1q_u16(output, this->_d);
    }

    simd operator+(const simd& other) const noexcept
    {
        return simd(vqaddq_u16(this->_d, other._d));
    }
    simd& operator+=(const simd& other) noexcept
    {
        this->_d = vqaddq_u16(this->_d, other._d);
        return *this;
    }
    simd operator-(const simd& other) const noexcept
    {
        return simd(vqsubq_u16(this->_d, other._d));
    }
    simd& operator-=(const simd& other) noexcept
    {
        this->_d = vqsubq_u16(this->_d, other._d);
        return *this;
    }
    simd operator*(const simd& other) const noexcept
    {
        return simd(vmulq_u16(this->_d, other._d));
    }
    simd& operator*=(const simd& other) noexcept
    {
        this->_d = vmulq_u16(this->_d, other._d);
        return *this;
    }
    uint16_t sum() const noexcept
    {
        uint16x4_t res = vadd_u16(vget_low_u16(this->_d), vget_high_u16(this->_d));
        res = vpadd_u16(res, res);
        return vget_lane_u16(res, 0) + vget_lane_u16(res, 1);
    }

private:
    uint16x8_t _d;
};

using simdu16x8 = simd<uint16_t, 8>;
