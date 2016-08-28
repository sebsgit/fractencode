#include "gaussian5x5.h"
#include "sampler.h"
#include <iostream>

using namespace Frac;

static constexpr const float _kernel[] = {
    1.f,  4.f,  7.f,  4.f, 1.f,
    4.f, 16.f, 26.f, 16.f, 4.f,
    7.f, 26.f, 41.f, 26.f, 7.f,
    4.f, 16.f, 26.f, 16.f, 4.f,
    1.f,  4.f,  7.f,  4.f, 1.f
};

static constexpr const int _normalizationFactor = 273;

Image GaussianBlur5x5::process(const Image &image) const {
    AbstractBufferPtr<Image::Pixel> data = Buffer<Image::Pixel>::alloc(image.height() * image.stride());
    Image result(data, image.width(), image.height(), image.stride());
    auto resultPtr = data->get();
    const SamplerLinear sampler(image);
    const int w = image.width();
    const int h = image.height();
    for (int y=0 ; y<h ; ++y) {
        for (int x=0 ; x<w ; ++x) {
            float sum = 0.0f;
            for (int k=-2; k<=2 ; ++k) {
                for (int j=-2; j<=2 ; ++j) {
                    const float p = (x+k<0 || x+k>=w || y+j<0 || y+j>=h) ? 0.0 : (1.0f*_kernel[k+2+(j+2)*5]*sampler(x+k, (y+j)));
                    sum += p;
                }
            }
            resultPtr[x + y*image.stride()] = convert<Image::Pixel>(sum / _normalizationFactor);
        }
    }
    return result;
}
