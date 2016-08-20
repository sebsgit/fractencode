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

static void test_encoder(const CmdArgs& args) {
    using namespace Frac;
    const Size32u gridSize(args.encoderParams.sourceGridSize, args.encoderParams.sourceGridSize);
    const GridPartitionCreator targetCreator(gridSize / 2, gridSize / 2);
    Timer timer;
    Image image(args.inputPath.c_str());
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
    result.savePng("result.png");
    std::cout << "decode stats: " << stats.iterations << " steps, rms: " << stats.rms << "\n";
}

int main(int argc, char *argv[])
{
    test_partition();
    if (argc > 1) {
        Frac::Image image(argv[1]);
        if (image.data()) {
            test_encoder(CmdArgs(argc, argv));
            Frac::Image corner = image.slice(image.width() / 2, image.height() / 2, image.width() / 2, image.height() / 2);
            Frac::Painter painter(corner);
            for (uint32_t i=0 ; i<corner.height() ; ++i)
                painter.set(i, i, 255);
            //corner.savePng("test.png");
            Frac::Transform t;
            t.setType(Frac::Transform::Flip_Rotate_270);
           // t.resize(corner, t.map(Frac::Size32u(corner.width() * 2, corner.height() * 2)), Frac::Transform::NearestNeighbor).savePng("resNN.png");
           // t.resize(corner, t.map(Frac::Size32u(corner.width() * 2, corner.height() * 2)), Frac::Transform::Bilinear).savePng("resBI.png");
            std::cout << Frac::RootMeanSquare().distance(image, image) << '\n';
        }
    }
    return 0;
}
