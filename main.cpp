#include "image/image.h"
#include "encode/encoder.h"
#include "transform.h"
#include "metrics.h"
#include "partition.h"
#include "utils/timer.h"
#include <iostream>
#include <cassert>
#include <cstring>

class CmdArgs {
public:
    std::string inputPath;
    Frac::Encoder::encode_parameters_t encoderParams;
    int decodeSteps = -1;
    bool color = false;

    CmdArgs(int argc, char** argv) {
        assert(argc > 1);
        this->inputPath = argv[1];
        this->_parse(argv + 2, argc - 2);
    }
private:
    void _parse(char** s, const int count) {
        int index = 0;
        while (index < count) {
            std::string tmp(s[index]);
            if (tmp == "--decode") {
                decodeSteps = atoi(s[index + 1]);
                ++index;
            } else if (tmp == "--grid") {
                encoderParams.sourceGridSize = atoi(s[index + 1]);
                ++index;
            } else if (tmp == "--rms") {
                encoderParams.rmsThreshold = atof(s[index + 1]);
                ++index;
            } else if (tmp == "--smax") {
                encoderParams.sMax = atof(s[index + 1]);
                ++index;
            } else if (tmp == "--color") {
                color = true;
            }
            ++index;
        }
    }
};

static void test_partition() {
    using namespace Frac;
    const uint32_t w = 128, h = 128;
    const uint32_t gridSize = 8;
    AbstractBufferPtr<Image::Pixel> buffer = Buffer<Image::Pixel>::alloc(w * h);
    Image image = Image(buffer, w, h, w);
    GridPartitionCreator gridCreator(Size32u(gridSize, gridSize), Size32u(gridSize, gridSize));
    Partition grid = gridCreator.create(image);
    assert(grid.size() == (w * h) / (gridSize * gridSize));
    uint8_t color = 0;
    for (auto it : grid) {
        Painter p(it->image());
        p.fill(color++);
    }
    //image.savePng("grid.png");
}

static Frac::Image encode_image(const CmdArgs& args, Frac::Image image) {
    using namespace Frac;
    const Size32u gridSize(args.encoderParams.sourceGridSize, args.encoderParams.sourceGridSize);
    const GridPartitionCreator targetCreator(gridSize / 2, gridSize / 2);
    Timer timer;
    timer.start();
    Encoder encoder(image, args.encoderParams, targetCreator);
    std::cout << "encoded in " << timer.elapsed() << " s.\n";
    auto data = encoder.data();
    uint32_t w = image.width(), h = image.height();
    AbstractBufferPtr<Image::Pixel> buffer = Buffer<Image::Pixel>::alloc(w * h);
    buffer->memset(0);
    Image result = Image(buffer, w, h, w);
    timer.start();
    Decoder decoder(result, args.decodeSteps);
    auto stats = decoder.decode(data);
    std::cout << "decoded in " << timer.elapsed() << " s.\n";
    std::cout << "decode stats: " << stats.iterations << " steps, rms: " << stats.rms << "\n";
    return result;
}

static void test_encoder(const CmdArgs& args) {
    using namespace Frac;
    if (args.color == false) {
        Image image(args.inputPath.c_str());
        Image result = encode_image(args, image);
        result.savePng("result.png");
    } else {
        PlanarImage image(args.inputPath.c_str());
        Image y = encode_image(args, image.y());
        Image u = encode_image(args, image.u());
        Image v = encode_image(args, image.v());
        PlanarImage result(y, u, v);
        result.savePng("result.png");
    }
}

static void test_statistics() {
    using namespace Frac;
    int w = 128;
    int h = 128;
    double imageSum = 0.0;
    auto buffer = Buffer<Image::Pixel>::alloc(w * h);
    for (int i=0 ; i<h ; ++i)
        for (int j=0 ; j<w ; ++j) {
            buffer->get()[j + h*i] = i + j;
            imageSum += i + j;
        }
    Image image(buffer, w, h, w);
    double testSum = ImageStatistics::sum(image);
    if (fabs(testSum - imageSum) > 0.001) {
        std::cout << "expected " << imageSum << ", actual " << testSum << '\n';
        exit(0);
    }
}

int main(int argc, char *argv[])
{
    test_statistics();
    test_partition();
    if (argc > 1) {
        Frac::Image image(argv[1]);
        if (image.data()) {
            test_encoder(CmdArgs(argc, argv));
        }
    }
    return 0;
}
