#ifdef FRAC_TESTS

#include "catch.hpp"
#include "encode/TransformEstimator2.hpp"
#include "encode/Classifier2.hpp"
#include <unordered_map>

using namespace Frac2;

TEST_CASE("TransformEstimator", "[encode][estimator]")
{
    SECTION("estimate simple")
    {
        ImagePlane source({ 8, 8 }, 8,
            { 1, 1, 2, 2, 40, 41, 50, 51,
            1, 1, 2, 2, 40, 41, 50, 51,
            3, 3, 4, 4, 70, 71, 80, 81,
            3, 3, 4, 4, 70, 71, 80, 81,
            10, 10, 10, 10, 0, 0, 0, 0,
            11, 11, 11, 11, 1, 1, 1, 1,
            10, 10, 10, 10, 0, 0, 0, 0,
            11, 11, 11, 11, 1, 1, 1, 1 });

        ImagePlane target({ 4, 4 }, 4,
            {40, 50, 2, 4,
            70, 80, 1, 3,
            0, 0, 10, 10,
            1, 1, 11, 11 });

        std::unordered_map<Point2du, Point2du, Point2du::hash> expectedResults;
        expectedResults[{0, 0}] = { 4, 0 };
        expectedResults[{2, 0}] = { 0, 0 };
        expectedResults[{0, 2}] = { 4, 4 };
        expectedResults[{2, 2}] = { 0, 4 };

        auto matcher = std::make_unique<TransformMatcher>(Frac::RootMeanSquare(), 0.0, 100.0);
        auto sourceGrid = Frac2::createUniformGrid(source.size(), Size32u(4, 4), Size32u(2, 2));
        TransformEstimator2 estimator(source, target, std::make_unique<DummyClassifier>(source, target), std::move(matcher), sourceGrid);
        auto targetGrid = Frac2::createUniformGrid(target.size(), Size32u(2, 2), Size32u(2, 2));
        for (const auto& targetPatch : targetGrid.items()) {
            auto result = estimator.estimate(targetPatch);
            auto it = expectedResults.find(targetPatch.origin);
            REQUIRE(it != expectedResults.end());
            REQUIRE(it->second == Point2du{result.x, result.y});
            expectedResults.erase(it);
        }
        REQUIRE(expectedResults.empty());
    }
}
#endif