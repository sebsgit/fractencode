#ifdef FRAC_TESTS
#include "image/ImageStatistics.hpp"
#include "catch.hpp"

using namespace Frac2;

static ImagePlane createTestImage(size_t size)
{
    const uint32_t padding = 32;
    const uint32_t stride = size + padding;
    ImagePlane result(Size32u(size, size), stride);
    for (size_t y = 0; y < size; ++y) {
        for (size_t x = 0; x < size; ++x) {
            result.setValue(x, y, y + 1);
        }
    }
    return result;
}

static ImagePlane createTestImage(size_t size, uint8_t value)
{
    const uint32_t padding = 32;
    const uint32_t stride = size + padding;
    ImagePlane result(Size32u(size, size), stride);
    for (size_t y = 0; y < size; ++y) {
        for (size_t x = 0; x < size; ++x) {
            result.setValue(x, y, value);
        }
    }
    return result;
}

TEST_CASE("ImageStatistics", "[image]")
{
    SECTION("sum")
    {
        const std::array<size_t, 6> sizes {2, 4, 8, 16, 32, 64};
        for (size_t i = 0; i < sizes.size(); ++i) {
            const UniformGridItem patch{Point2du(), Size32u(sizes[i], sizes[i])};
            const auto image = createTestImage(sizes[i]);
            const auto sum = ImageStatistics2::sum<float>(image, patch);
            const auto expectedSum = (sizes[i] * (1 + sizes[i]) / 2) * sizes[i];
            REQUIRE(sum == Approx(expectedSum));

            const auto maxImage = createTestImage(sizes[i], std::numeric_limits<uint8_t>::max());
            const auto maxSum = ImageStatistics2::sum<float>(maxImage, patch);
            const auto expectedMaxSum = std::numeric_limits<uint8_t>::max() * sizes[i] * sizes[i];
            REQUIRE(maxSum == Approx(expectedMaxSum));
        }
    }
    SECTION("mean")
    {
        auto image = createTestImage(4, 16);
        auto mean1x1 = ImageStatistics2::mean(image, UniformGridItem(Point2du(), Size32u(1, 1)));
        REQUIRE(mean1x1 == 16);
        auto mean2x2 = ImageStatistics2::mean(image, UniformGridItem(Point2du(), Size32u(4, 4)).topRight());
        REQUIRE(mean2x2 == 16);
    }
}

#endif
