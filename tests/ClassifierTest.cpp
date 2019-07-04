#ifdef FRAC_TESTS
#include "catch.hpp"
#include "encode/Classifier2.hpp"
#include "image/Image2.hpp"
#include "image/ImageIO.hpp"

#include <unordered_map>
#include <vector>

using namespace Frac2;

TEST_CASE("Classifier", "[encode]")
{
    const auto planes = ImageIO::loadImage("tests/input/lenna512x512.png");

    SECTION("categories")
    {
        struct test_data {
            int x = 0;
            int y = 0;
            int category = -1;
        };
        std::unordered_map<int, std::vector<test_data>> expectedData;
        expectedData[2] = std::vector<test_data> {
            { 204, 78, 0 }, { 242, 242, 1 }, { 6, 6, 2 }, { 82, 226, 3 }, { 418, 486, 4 }, { 384, 250, 5 }, { 136, 40, -1 }
        };
        expectedData[4] = std::vector<test_data> {
            { 416, 336, 5 }, { 440, 336, 0 }, { 448, 336, 1 }, { 504, 336, 2 }, { 316, 340, 3 }, { 336, 340, 4 }, { 400, 340, -1 }
        };
        expectedData[8] = std::vector<test_data> {
            { 184, 96, 0 }, { 192, 96, 1 }, { 264, 96, 2 }, { 368, 96, 3 }, { 400, 96, 4 }, { 440, 96, 5 }, { 472, 96, -1 }
        };
        expectedData[16] = std::vector<test_data> {
            { 320, 224, 4 }, { 80, 240, 5 }, { 416, 256, -1 }, { 464, 256, 0 }, { 0, 272, 1 }, { 96, 272, 2 }, { 112, 272, 3 }
        };
        expectedData[32] = std::vector<test_data> {
            { 384, 224, -1 }, { 448, 224, 0 }, { 0, 256, 1 }, { 96, 256, 2 }, { 160, 256, 3 }, { 288, 256, 4 }, { 64, 320, 5 }
        };
        expectedData[64] = std::vector<test_data> {
            { 64, 0, 0 }, { 192, 64, 1 }, { 448, 128, 2 }, { 256, 192, 3 }, { 256, 256, 4 }, { 128, 320, 5 }
        };

        for (const auto& d : expectedData) {
            const size_t size = d.first;
            const auto& vec = d.second;
            for (const auto& it : vec) {
                const UniformGridItem patch(Point2du(it.x, it.y), Size32u(size, size));
                auto category = BrightnessBlocksClassifier2::getCategory(planes[0], patch);
                REQUIRE(category == it.category);
            }
        }
    }
}

#endif
