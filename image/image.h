#ifndef FRAC_IMAGE_H
#define FRAC_IMAGE_H

#include "thirdparty/stb_image/stb_image.h"
#include "thirdparty/stb_image/stb_image_write.h"
#include "buffer.hpp"
#include "size.hpp"
#include <cstring>
#include <unordered_map>

namespace Frac {

//class ImageDataEntry

class ImageData {
public:
    static const int KeySum = 0;
    static const int KeyMean = 1;
    static const int KeyVariance = 2;
    static const int KeyBlockTypeBrightness = 3;

    void put(int key, double value) {
        _data[key] = value;
    }
    double get(const int key, const double defaultValue = -1.0) const {
        const auto it = _data.find(key);
        if (it != _data.end())
            return it->second;
        return defaultValue;
    }
private:
    std::unordered_map<int, double> _data;
};

class Image {
public:
    typedef uint16_t Pixel;
    Image(const char* fileName, const int channelCount = 1) {
        assert(channelCount == 1 && "multiple channels not implemented");
        int components = 0;
        int w, h;
        unsigned char* data = stbi_load(fileName, &w, &h, &components, channelCount);
        if (data) {
            _width = w;
            _height = h;
            _stride = _width * channelCount;
            if (sizeof(Pixel) > 1) {
                Pixel* buffer = (Pixel*)malloc(sizeof(Pixel) * _height * _stride);
                try {
                    for (size_t i=0 ; i<_height * _stride ; ++i) {
                        buffer[i] = convert<Pixel>(data[i]);
                    }
                } catch (...) {}
                free(data);
                data = (unsigned char*)buffer;
            }
            _data.reset(new Buffer<Pixel>((Pixel*)data, _height * _stride * sizeof(Pixel)));
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
        if (sizeof(Pixel) == 1)
            stbi_write_png(path, _width, _height, 1, _data->get(), _stride);
        else {
            auto buffer = convert<uint8_t>(_data);
            stbi_write_png(path, _width, _height, 1, buffer->get(), _stride);
        }
    }
    void map(const std::function<void(uint32_t, uint32_t)>& f) const {
        for (uint32_t y = 0 ; y<this->height() ; ++y) {
            for (uint32_t x = 0 ; x<this->width() ; ++x) {
                f(x, y);
            }
        }
    }
    void map(const std::function<void(Image::Pixel)>& f) const {
        for (uint32_t y = 0 ; y<this->height() ; ++y) {
            for (uint32_t x = 0 ; x<this->width() ; ++x) {
                f(this->_data->get()[x + y * stride()]);
            }
        }
    }
    ImageData& cache() const {
        return _cache;
    }
private:
    AbstractBufferPtr<Pixel> _data;
    uint32_t _width = 0, _height = 0, _stride = 0;
    mutable ImageData _cache;
};

class ImageStatistics {
public:
    static double sum(const Image& a) noexcept;
    static double mean(const Image& image) noexcept;
    static double variance(const Image& image) noexcept;
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
