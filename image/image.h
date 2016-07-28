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
    Image(const char* fileName, const int channelCount = 1) {
        assert(channelCount == 1 && "multiple channels not implemented");
        int components = 0;
        int w, h;
        unsigned char* data = stbi_load(fileName, &w, &h, &components, channelCount);
        if (data) {
            _width = w;
            _height = h;
            _stride = _width * channelCount;
            _data.reset(new Buffer<uint8_t>(data, _height * _stride));
        }
    }
    Image(AbstractBufferPtr<uint8_t> data, uint32_t width, uint32_t height, uint32_t stride)
        :_data(data)
        ,_width(width)
        ,_height(height)
        ,_stride(stride)
    {

    }

    uint32_t width() const { return _width; }
    uint32_t height() const { return _height; }
    uint32_t stride() const { return _stride; }
    AbstractBufferPtr<uint8_t> data() {
        return _data;
    }
    const AbstractBufferPtr<uint8_t> data() const {
        return _data;
    }
    const Size32u size() const noexcept {
        return Size32u(this->_width, this->_height);
    }
    Image slice(uint32_t x, uint32_t y, uint32_t width, uint32_t height) const {
        assert(x + width <= _width);
        assert(y + height <= _height);
        const auto offset = y * _stride + x;
        auto buffer = BufferSlice<uint8_t>::slice(_data, offset, _data->size() - offset);
        return Image(buffer, width, height, _stride);
    }
    uint8_t value(const uint32_t x, const uint32_t y) const {
        return _data->get()[x + y * stride()];
    }
    uint8_t value_2x2(uint32_t x, uint32_t y) const {
        if (x == width() - 1)
            --x;
        if (y == height() - 1)
            --y;
        const double total = u2d(value(x, y)) + u2d(value(x + 1, y)) + u2d(value(x, y + 1)) + u2d(value(x + 1, y + 1));
        return (uint8_t)(total / 4);
    }
    Image copy() const {
        return Image(this->_data->clone() , width(), height(), stride());
    }
    void savePng(const char* path) const {
        stbi_write_png(path, _width, _height, 1, _data->get(), _stride);
    }
private:
    AbstractBufferPtr<uint8_t> _data;
    uint32_t _width = 0, _height = 0, _stride = 0;
};

class Painter {
public:
    Painter(Image& image)
        :_image(image)
    {

    }
    void set(const uint32_t x, const uint32_t y, const uint8_t color) {
        _image.data()->get()[x + y * _image.stride()] = color;
    }
    void fill(const uint8_t color) {
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
