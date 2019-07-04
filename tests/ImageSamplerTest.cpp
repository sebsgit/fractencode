#ifdef FRAC_TESTS

#include "catch.hpp"
#include "image/Image2.hpp"
#include "image/sampler.h"

using namespace Frac2;

TEST_CASE("ImageSampler", "[image]")
{
    SECTION("bilinear")
    {
        ImagePlane image({ 8, 8 }, 8,
            { 1, 1, 2, 2, 3, 3, 4, 4,
            5, 5, 6, 6, 7, 7, 8, 8,
            9, 9, 10, 10, 11, 11, 12, 12,
            13, 13, 14, 14, 15, 15, 16, 16,
            17, 17, 18, 18, 19, 19, 20, 20,
            21, 21, 22, 22, 23, 23, 24, 24,
            25, 25, 26, 26, 27, 27, 28, 28,
            29, 29, 30, 30, 31, 31, 32, 32 });

        //
        auto sample2x2 = [](const auto& image, int32_t x, int32_t y, Transform t) {
            GridItemBase patch{Point2du(x, y), Size32u(2, 2)};
            return SamplerBilinear::sample<float>(image, patch, 0, 0, t);
        };
        auto sample4x4 = [](const auto& image, int32_t x, int32_t y, Transform t) {
            GridItemBase patch{ Point2du(x, y), Size32u(4, 4)};
            return SamplerBilinear::sample<float>(image, patch, 0, 0, t);
        };
        REQUIRE(sample2x2(image, 0, 0, Transform::Id) == (1 + 1 + 5 + 5) / 4.0f);
        REQUIRE(sample2x2(image, 1, 0, Transform::Id) == (1 + 2 + 5 + 6) / 4.0f);
        REQUIRE(sample2x2(image, 3, 3, Transform::Id) == (14 + 15 + 18 + 19) / 4.0f);
        REQUIRE(sample2x2(image, 3, 6, Transform::Id) == (26 + 27 + 30 + 31) / 4.0f);

        REQUIRE(sample4x4(image, 0, 0, Transform::Id) == (1 + 1 + 5 + 5) / 4.0f);
        REQUIRE(sample4x4(image, 0, 0, Transform::Rotate_270) == (2 + 2 + 6 + 6) / 4.0f);
        REQUIRE(sample4x4(image, 0, 0, Transform::Flip) == (9 + 9 + 13 + 13) / 4.0f);

        REQUIRE(sample4x4(image, 3, 4, Transform::Id) == (18 + 19 + 22 + 23) / 4.0f);
        REQUIRE(sample4x4(image, 3, 4, Transform::Rotate_90) == (26 + 27 + 30 + 31) / 4.0f);
        REQUIRE(sample4x4(image, 3, 4, Transform::Rotate_180) == (27 + 28 + 31 + 32) / 4.0f);
        REQUIRE(sample4x4(image, 3, 4, Transform::Rotate_270) == (19 + 20 + 23 + 24) / 4.0f);
        REQUIRE(sample4x4(image, 3, 4, Transform::Flip) == (26 + 27 + 30 + 31) / 4.0f);
    }
}
#endif