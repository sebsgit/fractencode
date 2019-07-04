#pragma once

#include "../simd_base.hpp"

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

template <>
class simd<uint16_t, 8> : public simd_common<simd<uint16_t, 8>> {
public:
    using type = uint16_t;
    static constexpr size_t value_count = 8;

    simd() noexcept
        : _d(_mm_setzero_si128())
    {
    }
    explicit simd(uint16_t value) noexcept
        : _d(_mm_set1_epi16(value)) //TODO: avoid conversion to short
    {
    }
    simd(uint16_t s0, uint16_t s1, uint16_t s2, uint16_t s3,
        uint16_t s4, uint16_t s5, uint16_t s6, uint16_t s7) noexcept
        : _d(_mm_setr_epi16(s0, s1, s2, s3, s4, s5, s6, s7)) //TODO: avoid conversion to short
    {
    }
    explicit simd(__m128i value) noexcept
        : _d(value)
    {
    }
    explicit simd(const uint16_t* input) noexcept
        : _d(is_aligned(input, 16) ? _mm_load_si128(reinterpret_cast<const __m128i*>(input))
            : _mm_loadu_si128(reinterpret_cast<const __m128i*>(input)))
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
        : _d(_mm_unpacklo_epi8(_mm_loadu_si64(input), _mm_setzero_si128()))
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
        _mm_store_si128(reinterpret_cast<__m128i*>(output), this->_d);
    }

    simd operator+(const simd& other) const noexcept
    {
        return simd(_mm_adds_epu16(this->_d, other._d));
    }
    simd& operator+=(const simd& other) noexcept
    {
        this->_d = _mm_adds_epu16(this->_d, other._d);
        return *this;
    }
    simd operator-(const simd& other) const noexcept
    {
        return simd(_mm_subs_epu16(this->_d, other._d));
    }
    simd& operator-=(const simd& other) noexcept
    {
        this->_d = _mm_subs_epu16(this->_d, other._d);
        return *this;
    }
    simd operator*(const simd& other) const noexcept
    {
        return simd(_mm_mullo_epi16(this->_d, other._d));
    }
    simd& operator*=(const simd& other) noexcept
    {
        this->_d = _mm_mullo_epi16(this->_d, other._d);
        return *this;
    }
    uint16_t sum() const noexcept
    {
        //   0, 1   2,3   4,5,   6,7
        // + 2, 3   0,1   6,7    4,5
        __m128i result = _mm_adds_epu16(this->_d, _mm_shuffle_epi32(this->_d, _MM_SHUFFLE(2, 3, 0, 1)));
        // + 4,5    6,7   0,1,   2,3
        result = _mm_adds_epu16(result, _mm_shuffle_epi32(this->_d, _MM_SHUFFLE(1, 0, 3, 2)));
        // + 6,7    4,5  2,3,   0,1
        result = _mm_adds_epu16(result, _mm_shuffle_epi32(this->_d, _MM_SHUFFLE(0, 1, 2, 3)));
        return static_cast<uint16_t>(_mm_extract_epi16(result, 0) + _mm_extract_epi16(result, 1));
    }

private:
    __m128i _d;
};

using simdu16x8 = simd<uint16_t, 8>;
