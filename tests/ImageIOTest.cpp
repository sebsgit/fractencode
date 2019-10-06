#ifdef FRAC_TESTS

#include "catch.hpp"
#include "image/ImageIO.hpp"

using namespace Frac2;

TEST_CASE("ImageIO", "[image]")
{
    SECTION("load image")
    {
        std::array<ImagePlane, 3> result = ImageIO::loadImage("tests/input/lenna512x512.png");
        REQUIRE(result[0].size() == Size32u(512, 512));
        REQUIRE(result[1].size() == Size32u(256, 256));
        REQUIRE(result[2].size() == Size32u(256, 256));
    }
    SECTION("load / save and compare")
    {
        std::array<ImagePlane, 3> result = ImageIO::loadImage("tests/input/lenna512x512.png");
        Image2<3> image(std::move(result));
        ImageIO::saveImage(image, "tests/output/lenna512_out.png");
    }
}
#endif
