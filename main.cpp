#include "image/image.h"
#include "encode/encoder.h"
#include "transform.h"
#include "metrics.h"
#include "partition.h"
#include <iostream>
#include <cassert>
#include <cstring>

static void test_partition() {
    using namespace Frac;
    const uint32_t w = 128, h = 128;
    const uint32_t gridSize = 8;
    AbstractBufferPtr<uint8_t> buffer = Buffer<uint8_t>::alloc(w * h);
    Image image = Image(buffer, w, h, w);
    GridPartition gridCreator = GridPartition(Size32u(gridSize, gridSize), Size32u(gridSize, gridSize));
    PartitionData grid = gridCreator.create(image);
    assert(grid.size() == (w * h) / (gridSize * gridSize));
    uint8_t color = 0;
    for (auto it : grid) {
        Painter p(it->image());
        p.fill(color++);
    }
    //image.savePng("grid.png");
}

static void test_encoder(const char* path) {
    using namespace Frac;
    Image image(path);
    Encoder encoder(image);
    auto data = encoder.data();
    uint32_t w = image.width(), h = image.height();
    AbstractBufferPtr<uint8_t> buffer = Buffer<uint8_t>::alloc(w * h);
    buffer->memset(0);
    Image result = Image(buffer, w, h, w);
    Decoder decoder(result, 10);
    decoder.decode(data);
    result.savePng("result.png");
}

int main(int argc, char *argv[])
{
    test_partition();
    if (argc > 1) {
        Frac::Image image(argv[1]);
        if (image.data()) {
            test_encoder(argv[1]);
            Frac::Image corner = image.slice(image.width() / 2, image.height() / 2, image.width() / 2, image.height() / 2);
            Frac::Painter painter(corner);
            for (uint32_t i=0 ; i<corner.height() ; ++i)
                painter.set(i, i, 255);
            //corner.savePng("test.png");
            Frac::Transform t;
            t.setType(Frac::Transform::Flip_Rotate_270);
            t.resize(corner, t.map(Frac::Size32u(corner.width() * 2, corner.height() * 2)), Frac::Transform::NearestNeighbor).savePng("resNN.png");
            t.resize(corner, t.map(Frac::Size32u(corner.width() * 2, corner.height() * 2)), Frac::Transform::Bilinear).savePng("resBI.png");
            std::cout << Frac::RootMeanSquare().distance(image, image) << '\n';
        }
    }
    return 0;
}
