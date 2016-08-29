#include "image/image.h"
#include "encode/encoder.h"
#include "transform.h"
#include "metrics.h"
#include "partition.h"
#include "partition/quadtreepartition.h"
#include "utils/timer.h"
#include "process/gaussian5x5.h"
#include "process/sobel.h"
#include <iostream>
#include <cassert>
#include <cstring>

class CmdArgs {
public:
    std::string inputPath;
    Frac::Encoder::encode_parameters_t encoderParams;
    int decodeSteps = -1;
    bool color = false;
    bool useQuadtree = false;

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
            } else if (tmp == "--source") {
                encoderParams.sourceGridSize = atoi(s[index + 1]);
                ++index;
            } else if (tmp == "--target") {
                encoderParams.targetGridSize = atoi(s[index + 1]);
                ++index;
            } else if (tmp == "--rms") {
                encoderParams.rmsThreshold = atof(s[index + 1]);
                ++index;
            } else if (tmp == "--smax") {
                encoderParams.sMax = atof(s[index + 1]);
                ++index;
            } else if (tmp == "--color") {
                color = true;
            } else if (tmp == "--quadtree") {
                useQuadtree = true;
            }
            ++index;
        }
        if (encoderParams.targetGridSize >= encoderParams.sourceGridSize || encoderParams.targetGridSize < 2 || encoderParams.sourceGridSize < 2) {
            std::cout << "invalid source / target size\n";
            throw std::exception();
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
    PartitionPtr grid = gridCreator.create(image);
    assert(grid->size() == (w * h) / (gridSize * gridSize));
    uint8_t color = 0;
    for (auto it : *grid) {
        Painter p(it->image());
        p.fill(color++);
    }
    //image.savePng("grid.png");
}

static Frac::Image encode_image(const CmdArgs& args, Frac::Image image) {
    using namespace Frac;
    const Size32u gridSize(args.encoderParams.targetGridSize, args.encoderParams.targetGridSize);
    std::unique_ptr<PartitionCreator> targetCreator;
    if (args.useQuadtree)
        targetCreator.reset(new QuadtreePartitionCreator(Size32u(args.encoderParams.sourceGridSize, args.encoderParams.sourceGridSize) / 2, gridSize));
    else
        targetCreator.reset(new GridPartitionCreator(gridSize, gridSize));
    Timer timer;
    timer.start();
    Encoder encoder(image, args.encoderParams, *targetCreator);
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
    Timer timer;
    timer.start();
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
    std::cout << "total time: " << timer.elapsed() << " s.\n";
}

static void test_statistics() {
    using namespace Frac;
    int w = 64;
    int h = 64;
    double imageSum = 0.0;
    uint32_t stride = w + 64;
    auto buffer = Buffer<Image::Pixel>::alloc(stride * h);
    for (int i=0 ; i<h ; ++i)
        for (int j=0 ; j<w ; ++j) {
            buffer->get()[j + stride*i] = i + j;
            imageSum += i + j;
        }
    const double mean = imageSum / (w * h);
    double variance = 0.0;
    for (int i=0 ; i<h ; ++i)
        for (int j=0 ; j<w ; ++j) {
            const auto p = buffer->get()[j + stride*i];
            variance += (p - mean) * (p - mean);
        }
    variance /= (w * h);
    Image image(buffer, w, h, stride);
    const double testSum = ImageStatistics::sum(image);
    if (fabs(testSum - imageSum) > 0.001) {
        std::cout << "expected sum " << imageSum << ", actual " << testSum << '\n';
        exit(0);
    }
    const double testVariance = ImageStatistics::variance(image);
    if (fabs(testVariance - variance) > 0.001) {
        std::cout << "expected variance " << variance << ", actual " << testVariance << '\n';
        exit(0);
    }
}

static void test_blur() {
    using namespace Frac;
    const uint8_t expected[] = {
        0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 8, 9, 9, 7,
        1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 10,
        1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 11,
        2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 15, 12,
        3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 16, 12,
        3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 17, 13,
        4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 18, 14,
        5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 19, 14,
        5, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 20, 15,
        6, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 21, 16,
        7, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 22, 17,
        8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 23, 17,
        8, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 24, 18,
        9, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 25, 19,
        9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 24, 18,
        7, 10, 11, 12, 12, 13, 14, 14, 15, 16, 17, 17, 18, 19, 18, 13
    };

    const int w = 16;
    const int h = 16;
    const uint32_t stride = w + 64;
    auto buffer = Buffer<Image::Pixel>::alloc(stride * h);
    for (int i=0 ; i<h ; ++i)
        for (int j=0 ; j<w ; ++j) {
            buffer->get()[j + stride*i] = i + j;
        }
    Image image(buffer, w, h, stride);
    GaussianBlur5x5 blur;
    auto blurred = blur.process(image);
    int k = 0;
    for (int i=0 ; i<h ; ++i) {
        for (int j=0 ; j<w ; ++j) {
            if (blurred.data()->get()[j + stride*i] != expected[k]) {
                std::cout << "gaussian blur error: " << j << ' ' << i << " - "<< (int)blurred.data()->get()[j + stride*i] << "!=" << (int)expected[k];
                exit(0);
            }
            ++k;
        }
    }
}

int main(int argc, char *argv[])
{
    test_statistics();
    test_partition();
    test_blur();
    if (argc > 1) {
        Frac::Image image(argv[1]);
        if (image.data()) {

            Frac::Image test = Frac::SobelOperator().process(image);
            test.savePng("sobel.png");

            test_encoder(CmdArgs(argc, argv));
        }
    }
    return 0;
}
