#ifndef FRAC_IMAGE_H
#define FRAC_IMAGE_H

#include "thirdparty/stb_image/stb_image.h"
#include "thirdparty/stb_image/stb_image_write.h"
#include "buffer.hpp"
#include "size.hpp"
#include <cstring>
#include <unordered_map>
#ifndef FRAC_NO_THREADS
#include <atomic>
#include <mutex>
#endif

namespace Frac {

class ImageData {
#ifndef FRAC_NO_THREADS
	using ValueType = std::atomic<double>;
#else
	using ValueType = double;
#endif
public:
	static const int KeySum = 0;
	static const int KeyMean = 1;
	static const int KeyVariance = 2;
	static const int KeyBlockTypeBrightness = 3;

	ImageData() {}

	void put(int key, double value) {
		_data[key] = value;
	}
	double get(const int key, const double defaultValue = -1.0) const {
		return (_data[key] != -1) ? (double)_data[key] : defaultValue;
	}
private:
	ValueType _data[KeyBlockTypeBrightness + 1] = {-1, -1, -1, -1};
};

class Image {
public:
	typedef uint8_t Pixel;

	enum CachePolicy {
		CacheFull,
		NoCache
	};

	Image(const char* fileName, const int channelCount = 1) {
		assert(channelCount == 1 && "multiple channels not implemented");
		int components = 0;
		int w, h;
		unsigned char* data = stbi_load(fileName, &w, &h, &components, channelCount);
		if (data) {
			_size.setX(w);
			_size.setY(h);
			_stride = w * channelCount;
			if (sizeof(Pixel) > 1) {
				Pixel* buffer = (Pixel*)malloc(sizeof(Pixel) * h * _stride);
				try {
					for (size_t i=0 ; i<h * _stride ; ++i) {
						buffer[i] = convert<Pixel>(data[i]);
					}
				} catch (...) {}
				free(data);
				data = (unsigned char*)buffer;
			}
			_data.reset(new Buffer<Pixel>((Pixel*)data, h * _stride * sizeof(Pixel), [](uint8_t* ptr) { ::free(ptr); }));
		}
	}
	Image(AbstractBufferPtr<Pixel> data, uint32_t width, uint32_t height, uint32_t stride, Image::CachePolicy cache = Image::CacheFull)
		:_data(data)
		,_size(width, height)
		,_stride(stride)
		,_cache(cache == CacheFull ? new ImageData : nullptr)
	{

	}
	Image(uint32_t width, uint32_t height, uint32_t stride)
		:_data(Buffer<Pixel>::alloc(height * stride))
		,_size(width, height)
		,_stride(stride)
	{

	}

	Image() {}

	uint32_t width() const { return _size.x(); }
	uint32_t height() const { return _size.y(); }
	uint32_t stride() const { return _stride; }
	Size32u size() const { return _size; }
	AbstractBufferPtr<Pixel> data() {
		return _data;
	}
	const AbstractBufferPtr<Pixel> data() const {
		return _data;
	}
	bool empty() const noexcept {
		return !_data;
	}
	Image slice(uint32_t x, uint32_t y, uint32_t w, uint32_t h, Image::CachePolicy cached = Image::CacheFull) const {
		assert(x + w <= width());
		assert(y + h <= height());
		const auto offset = y * _stride + x;
		auto buffer = BufferSlice<Pixel>::slice(_data, offset, _data->size() - offset);
		return Image(buffer, w, h, _stride, cached);
	}
	Image copy() const {
		return Image(this->_data->clone() , width(), height(), stride());
	}
	void savePng(const std::string& path) const {
		if (sizeof(Pixel) == 1)
			stbi_write_png(path.c_str(), width(), height(), 1, _data->get(), _stride);
		else {
			auto buffer = convert<uint8_t>(_data);
			stbi_write_png(path.c_str(), width(), height(), 1, buffer->get(), _stride);
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
	const std::shared_ptr<ImageData>& cache() const {
		return _cache;
	}
private:
	AbstractBufferPtr<Pixel> _data;
	Size32u _size;
	uint32_t _stride = 0;
	mutable std::shared_ptr<ImageData> _cache;
};

class PlanarImage {
public:
	explicit PlanarImage(const char* path) {
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
	void savePng(const std::string& path) {
		auto buffer = Buffer<uint8_t>::alloc(_y.width() * _y.height() * 3);
		this->packToRgb(buffer->get());
		stbi_write_png(path.c_str(), _y.width(), _y.height(), 3, buffer->get(), _y.width() * 3);
	}
private:
	void unpackRgb(const unsigned char* rgb, uint32_t width, uint32_t height, uint32_t stride) {
		const uint32_t yStride = width + 64;
		const uint32_t uvStride = (width / 2) + 64;
		auto yBuff = Buffer<Image::Pixel>::alloc(yStride * height);
		auto uBuff = Buffer<Image::Pixel>::alloc((uvStride * height) / 2);
		auto vBuff = Buffer<Image::Pixel>::alloc((uvStride * height) / 2);
		for (size_t y=0 ; y<height ; ++y) {
			for (size_t x=0 ; x<width ; ++x) {
				auto r = rgb[x * 3 + y * stride + 0];
				auto g = rgb[x * 3 + y * stride + 1];
				auto b = rgb[x * 3 + y * stride + 2];
				auto yp = 0.299 * r + 0.587 * g + 0.114 * b;
				auto up = -0.169 * r - 0.331 * g + 0.499 * b + 128;
				auto vp = 0.499 * r - 0.418 * g - 0.0813 * b + 128;
				yBuff->get()[x + y * yStride] = clamp(yp);
				uBuff->get()[(x / 2) + ((y / 2)  * uvStride)] = clamp(up);
				vBuff->get()[(x / 2) + ((y / 2) * uvStride)] = clamp(vp);
			}
		}
		_y = Image(yBuff, width, height, yStride);
		_u = Image(uBuff, width / 2, height / 2, uvStride);
		_v = Image(vBuff, width / 2, height / 2, uvStride);
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
	uint8_t clamp(const double x) const noexcept {
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
	explicit Painter(Image& image)
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
