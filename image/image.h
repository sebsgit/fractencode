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
    Image() {}

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
        auto ptr = this->_data->get();
        for (uint32_t y = 0 ; y<this->height() ; ++y) {
            for (uint32_t x = 0 ; x<this->width() ; ++x) {
                f(ptr[x + y * stride()]);
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

class PlanarImage {
public:
    PlanarImage(const char* path) {
        int components = 0;
        int w, h;
        unsigned char* data = stbi_load(path, &w, &h, &components, 3);
        unpackRgb(data, w, h, w * 3);
        free(data);
    }
    PlanarImage(Image x, Image y, Image z)
        :_y(x), _u(y), _v(z)
    {

    }
    Image y() const noexcept {
        return _y;
    }
    Image u() const noexcept {
        return _u;
    }
    Image v() const noexcept {
        return _v;
    }
    void savePng(const char* path) {
        auto buffer = Buffer<uint8_t>::alloc(_y.width() * _y.height() * 3);
        this->packToRgb(buffer->get());
        stbi_write_png(path, _y.width(), _y.height(), 3, buffer->get(), _y.width() * 3);
    }
private:
    void unpackRgb(const unsigned char* rgb, uint32_t width, uint32_t height, uint32_t stride) {
        auto yBuff = Buffer<Image::Pixel>::alloc(width * height);
        auto uBuff = Buffer<Image::Pixel>::alloc((width * height) / 4);
        auto vBuff = Buffer<Image::Pixel>::alloc((width * height) / 4);
        for (size_t y=0 ; y<height ; ++y) {
            for (size_t x=0 ; x<width ; ++x) {
                auto r = rgb[x * 3 + y * stride + 0];
                auto g = rgb[x * 3 + y * stride + 1];
                auto b = rgb[x * 3 + y * stride + 2];
                auto yp = 0.299 * r + 0.587 * g + 0.114 * b;
                auto up = -0.169 * r - 0.331 * g + 0.499 * b + 128;
                auto vp = 0.499 * r - 0.418 * g - 0.0813 * b + 128;
                yBuff->get()[x + y * width] = clamp(yp);
                uBuff->get()[(x / 2) + ((y / 2)  * (width / 2))] = clamp(up);
                vBuff->get()[(x / 2) + ((y / 2) * (width / 2))] = clamp(vp);
            }
        }
        _y = Image(yBuff, width, height, width);
        _u = Image(uBuff, width / 2, height / 2, width / 2);
        _v = Image(vBuff, width / 2, height / 2, width / 2);
    }
    void packToRgb(unsigned char* rgb) const {
        for (size_t y = 0 ; y<_y.height() ; ++y) {
            for (size_t x=0 ; x<_y.width() ; ++x) {
                unsigned char* ptr = rgb + (x * 3 + y * _y.stride() * 3);
                double yp = _y.data()->get()[x + y * _y.stride()];
                double up = _u.data()->get()[(x / 2) + (y / 2) * _u.stride()];
                double vp = _v.data()->get()[(x / 2) + (y / 2) * _v.stride()];
                ptr[0] = clamp(yp + 1.402 * (vp - 128));
                ptr[1] = clamp(yp - 0.344 * (up - 128) - 0.714 * (vp - 128));
                ptr[2] = clamp(yp + 1.772 * (up - 128));
            }
        }
    }
    uint8_t clamp(double x) const noexcept {
        return x < 0.0 ? 0 : x > 255 ? 255 : (uint8_t)x;
    }
private:
    Image _y, _u, _v;
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
