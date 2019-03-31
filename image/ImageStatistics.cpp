#include "ImageStatistics.hpp"
#include "utils/simd/simdu16x8.hpp"

uint32_t Frac2::ImageStatistics2::sum_u32(const ImagePlane & image, const GridItemBase & item) noexcept
{
    uint32_t result = 0;
    for (uint32_t y = 0; y < item.size.y(); ++y)
        for (uint32_t x = 0; x < item.size.x(); ++x)
            result += image.value(item.origin.x() + x, item.origin.y() + y);
    return result;
}

// for items less than or equal to {16 x 16}
uint16_t Frac2::ImageStatistics2::sum_u16(const ImagePlane & image, const GridItemBase & item) noexcept
{
    uint16_t result = 0;
    if (item.size.x() == 4 && item.size.y() == 4) {
        auto row0 = image.data() + item.origin.y() * image.stride() + item.origin.x();
        auto row1 = image.data() + (item.origin.y() + 1) * image.stride() + item.origin.x();
        auto row2 = image.data() + (item.origin.y() + 2) * image.stride() + item.origin.x();
        auto row3 = image.data() + (item.origin.y() + 3) * image.stride() + item.origin.x();
        simdu16x8 rows01(row0[0], row0[1], row0[2], row0[3], row1[0], row1[1], row1[2], row1[3]);
        rows01 += simdu16x8(row2[0], row2[1], row2[2], row2[3], row3[0], row3[1], row3[2], row3[3]);
        result = rows01.sum();
    }
    else if (item.size.x() == 8 && item.size.y() == 8)
    {
        auto row = image.data() + item.origin.y() * image.stride() + item.origin.x();
        simdu16x8 total(row);
        for (int y = 1; y < 8; ++y) {
            row += image.stride();
            total += simdu16x8(row);
        }
        result = total.sum();
    }
    else {
        FRAC_ASSERT(item.size.x() <= 16);
        for (uint32_t y = 0; y < item.size.y(); ++y)
            for (uint32_t x = 0; x < item.size.x(); ++x)
                result += image.value(item.origin.x() + x, item.origin.y() + y);
    }
    return result;
}
