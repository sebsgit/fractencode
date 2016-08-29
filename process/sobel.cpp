#include "process/sobel.h"
#include <cmath>

using namespace Frac;

static const int _kernel_x[] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

static const int _kernel_y[] = {
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1
};

Image SobelOperator::process(const Image &image) const {
    const auto buffer = this->calculate(image);
    auto imageData = Buffer<Image::Pixel>::alloc(image.width() * image.height());
    Image result(imageData, image.width(), image.height(), image.width());
    for (uint32_t y=0 ; y<image.height() ; ++y) {
        for (uint32_t x=0 ; x<image.width() ; ++x) {
            const result_t dv = buffer->get()[x + y * image.width()];
            const uint8_t gradient = (uint8_t)sqrt( dv.dx*dv.dx + dv.dy*dv.dy );
            imageData->get()[x + y * image.width()] = gradient;
        }
    }
    return result;
}

AbstractBufferPtr<SobelOperator::result_t> SobelOperator::calculate(const Image &image) const {
    auto result = Buffer<result_t>::alloc(image.width() * image.height());
    for (int y=0 ; y<(int)image.height() ; ++y) {
        for (int x=0 ; x<(int)image.width() ; ++x) {
            result_t derivative;
            for (int i=-1 ; i<=1 ; ++i) {
                for (int j=-1 ; j<=1 ; ++j) {
                    const int opx = _kernel_x[(j + 1) + (i + 1)*3];
                    const int opy = _kernel_y[(j + 1) + (i + 1)*3];
                    const int xs = x + j < 0 ? 0 : x + j >= (int)image.width() ? image.width() - 1 : x + j;
                    const int ys = y + i < 0 ? 0 : y + i >= (int)image.height() ? image.height() - 1 : y + i;
                    const auto value = image.data()->get()[xs + ys * image.stride()];
                    derivative.dx += (int)value * opx;
                    derivative.dy += (int)value * opy;
                }
            }
            result->get()[x + y * image.width()] = derivative;
        }
    }
    return result;
}
