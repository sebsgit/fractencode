#ifdef FRAC_TESTS

#include "catch.hpp"
#include "image/Image2.hpp"
#include "image/metrics.h"
#include "image/partition2.hpp"

using Frac::Point2du;
using Frac::Size32u;

TEST_CASE("Partition", "[image][partition]")
{
    SECTION("basic grid")
    {
        const Size32u imageSize(512, 512);
        const Size32u itemSize(32, 32);
        const Size32u itemOffset = itemSize;
        auto grid = Frac2::createUniformGrid<Frac2::UniformGridItem>(imageSize, itemSize, itemOffset);
        REQUIRE(grid.items().size() == (imageSize.x() / itemSize.x()) * (imageSize.y() / itemSize.y()));
    }
    SECTION("grid with data")
    {
        using GridElementType = Frac2::GridItem<int>;
        const Size32u imageSize(4, 8);
        const Size32u itemSize(2, 4);
        const Size32u itemOffset = itemSize;
        auto elementDataInit = [](const Point2du& p, const Size32u&) {
            return p.x() + p.y();
        };
        auto grid = Frac2::createUniformGrid<GridElementType>(imageSize, itemSize, itemOffset, elementDataInit);
        REQUIRE(grid.items().size() == 4);
        for (const auto& item : grid.items()) {
            REQUIRE(item.data == item.origin.x() + item.origin.y());
        }
    }
    SECTION("distance (same size)")
    {
        using namespace Frac2;
        Frac::RootMeanSquare metrics;
        ImagePlane imageA({ 16, 16 }, 16);
        ImagePlane imageB({ 16, 16 }, 16);
        for (int y = 0; y < 16; ++y) {
            for (int x = 0; x < 16; ++x) {
                imageA.data()[y * imageA.stride() + x] = y % 4;
                imageB.data()[y * imageB.stride() + x] = x % 4;
            }
        }
        auto grid = Frac2::createUniformGrid<UniformGridItem>(imageA.size(), { 4, 4 }, {4, 4});
        for (const auto& item : grid.items()) {
            auto d = metrics.distance(imageA, imageB, item, item);
            REQUIRE(d > 0);
        }
    }
}
#endif