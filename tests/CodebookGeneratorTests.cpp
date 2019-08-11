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
        const size_t dataSize = 2000;
        const size_t codewordCount = 10;
        const float rangeStart = -100.0f;
        const float rangeEnd = 100.0f;
        const float epsilon = 0.01f;
        std::vector<float> data;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution dist(rangeStart, rangeEnd);
        std::generate_n(std::back_inserter(data), dataSize, [&dist, &gen]() { return dist(gen); });
        REQUIRE(data.size() == dataSize);

        auto codebook = generateCodebook<float>(data.begin(), data.end(), codewordCount, epsilon, [](auto a, auto b) { return std::abs(a - b); });
        REQUIRE(codebook.size() == codewordCount);
        for (auto & c : codebook) {
            REQUIRE(c >= rangeStart);
            REQUIRE(c <= rangeEnd);
        }
    }
}

#endif
