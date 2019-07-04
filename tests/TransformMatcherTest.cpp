#ifdef FRAC_TESTS

#include "encode/transformmatcher.h"
#include "catch.hpp"
#include "image/Image2.hpp"

using namespace Frac2;

TEST_CASE("TransformMatcher", "[encode][matcher]")
{
    SECTION("basic match")
    {
        ImagePlane source({ 8, 8 }, 8,
            { 1, 1, 2, 2, 40, 41, 50, 51,
            1, 1, 2, 2, 40, 41, 50, 51,
            3, 3, 4, 4, 70, 71, 80, 81,
            3, 3, 4, 4, 70, 71, 80, 81,
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1 });

        ImagePlane target({ 4, 4 }, 4,
            { 2, 4, 40, 50,
            1, 3, 70, 80,
            0, 0, 0, 0,
            1, 1, 1, 1 });

        Frac::RootMeanSquare metric;
        Frac::TransformMatcher matcher(metric, 0.0, 100.0);
        auto score = matcher.match(source, UniformGridItem{ Point2du(0, 0), Size32u{4, 4} }, 
            target, UniformGridItem{ Point2du(0, 0), Size32u{2, 2} });
        REQUIRE(score.distance == Approx(0.0));
        REQUIRE(score.transform == Transform::Rotate_270);
        REQUIRE(score.contrast < 1.0);
        REQUIRE(score.brightness < 1.0);
    }
}
#endif