#pragma once

#include <array>
#include <iosfwd>
#ifndef __ANDROID__
extern "C" {
#include <emmintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
}
#endif

template <typename T, size_t ValueCount>
class simd;

//TODO: add CPUID checks
class simd_base {
public:
    enum cache_coherence {
        coherent,
        non_temporal
    };
    enum compare_flags {
        equal,
        lower,
        lower_equal,
        greater,
        greater_equal,
        not_equal
    };
    static void load_fence()
    {
#ifndef __ANDROID__
        //TODO: optional assert for SSE2 support
        _mm_lfence();
#endif
    }
    static void store_fence()
    {
#ifndef __ANDROID__
        _mm_sfence();
#endif
    }
    static bool is_aligned(const void* address, uint32_t alignment) noexcept
    {
        return reinterpret_cast<uint64_t>(address) % alignment == 0;
    }
};

template <typename SimdClass>
class simd_common : public simd_base {
public:
    static constexpr size_t bit_count() noexcept
    {
        return SimdClass::value_count * sizeof(typename SimdClass::type) * 8;
    }

    auto to_array() const noexcept
    {
        alignas(16) std::array<typename SimdClass::type, SimdClass::value_count> result;
        static_cast<const SimdClass*>(this)->store_aligned(result.data());
        return result;
    }

    friend std::ostream& operator<<(std::ostream& out, const simd_common& value)
    {
        const auto data = value.to_array();
        out << '[';
        for (size_t i = 0; i < data.size() - 1; ++i)
            out << data[i] << ',';
        return out << data[data.size() - 1] << ']';
    }
};
