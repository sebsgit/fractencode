#ifdef FRAC_TESTS
#include "catch.hpp"
#include "encode/CodebookGenerator.hpp"

using namespace Frac;

TEST_CASE("CodebookGenerator", "[encode]")
{
    SECTION("index generator")
    {
        const size_t maxIndex = 10;
        std::random_device rd;
        UniqueIndexGenerator generator(maxIndex, rd);
        REQUIRE(generator.countGenerated() == 0);
        std::set<size_t> indices;
        while (generator.countGenerated() <= maxIndex) {
            indices.insert(generator.next());
        }
        REQUIRE(indices.size() == maxIndex + 1);
        REQUIRE(generator.next() == std::numeric_limits<size_t>::max());
        generator.reset();
        REQUIRE(generator.countGenerated() == 0);
    }
    SECTION("basic float")
    {

    }
}

#endif
