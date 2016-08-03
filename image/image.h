#ifndef FRAC_IMAGE_H
#define FRAC_IMAGE_H

#include "thirdparty/stb_image/stb_image.h"
#include "thirdparty/stb_image/stb_image_write.h"
#include "buffer.hpp"
#include "size.hpp"
#include <cstring>

namespace Frac {

class Image {
public:
    typedef uint8_t Pixel;
    Image(const char* fileName, const int channelCount = 1) {
        assert(channelCount == 1 && "multiple channels not implemented");
        int components = 0;
        int w, h;
        unsigned char* data = stbi_load(fileName, &w, &h, &components, channelCount);
        if (data) {
            _width = w;
            _height = h;
            _stride = _width * channelCount;
            _data.reset(new Buffer<Pixel>(data, _height * _stride));
        }
    }
    Image(AbstractBufferPtr<Pixel> data, uint32_t width, uint32_t height, uint32_t stride)
        :_data(data)
        ,_width(width)
        ,_height(height)
        ,_stride(stride)
    {

    }

    uint32_t width() const { return _width; }
    uint32_t height() const { return _height; }
    uint32_t stride() const { return _stride; }
    AbstractBufferPtr<Pixel> data() {
        return _data;
    }
    const AbstractBufferPtr<Pixel> data() const {
        return _data;
    }
    const Size32u size() const noexcept {
        return Size32u(this->_width, this->_height);
    }
    Image slice(uint32_t x, uint32_t y, uint32_t width, uint32_t height) const {
        assert(x + width <= _width);
        assert(y + height <= _height);
        const auto offset = y * _stride + x;
        auto buffer = BufferSlice<Pixel>::slice(_data, offset, _data->size() - offset);
        return Image(buffer, width, height, _stride);
    }
    Image copy() const {
        return Image(this->_data->clone() , width(), height(), stride());
    }
    void savePng(const char* path) const {
        stbi_write_png(path, _width, _height, 1, _data->get(), _stride);
    }
    void map(const std::function<void(uint32_t, uint32_t)>& f) const {
        for (uint32_t y = 0 ; y<this->height() ; ++y) {
            for (uint32_t x = 0 ; x<this->width() ; ++x) {
                f(x, y);
            }
        }
    }
private:
    AbstractBufferPtr<Pixel> _data;
    uint32_t _width = 0, _height = 0, _stride = 0;
};

class SamplerLinear {
public:
    SamplerLinear(const Image& source)
        : _source(source.data()->get())
        , _stride(source.stride())
    {

    }
    Image::Pixel operator() (uint32_t x, uint32_t y) const {
        return _source[x + y * _stride];
    }
private:
    const Image::Pixel* _source;
    const uint32_t _stride;
};

class SamplerBilinear {
public:
    SamplerBilinear(const Image& source)
        : _source(source.data()->get())
        , _stride(source.stride())
        , _width(source.width())
        , _height(source.height())
    {

    }
    Image::Pixel operator() (uint32_t x, uint32_t y) const {
        if (x == _width - 1)
            --x;
        if (y == _height - 1)
            --y;
        const int total = (int)_source[x + y * _stride] + (int)_source[x + 1 + y * _stride] + (int)_source[x + (y + 1) * _stride] + (int)_source[x + 1 + (y + 1) * _stride];
        return (Image::Pixel)(total / 4);
    }
private:
    const Image::Pixel* _source;
    const uint32_t _stride;
    const uint32_t _width;
    const uint32_t _height;
};


class Painter {
public:
    Painter(Image& image)
        :_image(image)
    {

    }
    void set(const uint32_t x, const uint32_t y, const Image::Pixel color) {
        _image.data()->get()[x + y * _image.stride()] = color;
    }
    void fill(const Image::Pixel color) {
        for (uint32_t y = 0 ; y < _image.height() ; ++y) {
            for (uint32_t x = 0 ; x < _image.width() ; ++x) {
                this->set(x, y, color);
            }
        }
    }

private:
    Image& _image;
};

}

#endif
