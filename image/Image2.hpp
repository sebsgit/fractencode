#pragma once

#include "utils/attributes.hpp"
#include "utils/point2d.hpp"
#include "utils/size.hpp"

#include <array>
#include <cstdlib>
#include <vector>
#include <memory>
#include <cstring>

namespace Frac2 {
using namespace Frac;

template <typename T, size_t alignment>
struct AlignedBuffer {
    static_assert(std::is_pod<T>::value, "non-pod data in aligned buffer");

public:
    std::unique_ptr<T, decltype(::_aligned_free)*> data;
    size_t count = 0;

    AlignedBuffer()
        : data(nullptr, &_aligned_free)
    {
    }

    explicit AlignedBuffer(size_t elementCount)
        : data(static_cast<T*>(_aligned_malloc(elementCount * sizeof(T), alignment)), &_aligned_free)
        , count(elementCount)
    {
    }

    explicit AlignedBuffer(std::initializer_list<T> values)
        : data(static_cast<T*>(_aligned_malloc(values.size() * sizeof(T), alignment)), &_aligned_free)
        , count(values.size())
    {
        size_t i = 0;
        for (auto& x : values)
            this->data.get()[i++] = x;
    }

    explicit AlignedBuffer(const std::vector<T>& values)
        : data(static_cast<T*>(_aligned_malloc(values.size() * sizeof(T), alignment)), &_aligned_free)
        , count(values.size())
    {
        size_t i = 0;
        for (auto& x : values)
            this->data.get()[i++] = x;
    }

    AlignedBuffer clone() const
    {
        AlignedBuffer result(this->count);
        std::memcpy(result.data.get(), this->data.get(), this->count * sizeof(T));
        return result;
    }

    T operator[](size_t index) const noexcept
    {
        return this->data.get()[index];
    }

    T& operator[](size_t index) noexcept
    {
        return this->data.get()[index];
    }
};

class ImagePlane {
public:
    FRAC_NO_COPY(ImagePlane);

    ImagePlane(const Size32u& size, uint32_t stride)
        : _data(size.y() * stride)
        , _stride(stride)
        , _size(size)
    {
    }
    ImagePlane(const Size32u& size, uint32_t stride, std::initializer_list<uint8_t> initialData)
        : _data(initialData)
        , _stride(stride)
        , _size(size)
    {
    }
    ImagePlane(const Size32u& size, uint32_t stride, std::vector<uint8_t>&& initialData)
        : _data(initialData)
        , _stride(stride)
        , _size(size)
    {
    }
    ImagePlane() {}
    ImagePlane(ImagePlane&&) = default;
    ImagePlane& operator=(ImagePlane&&) = default;
    auto size() const noexcept { return this->_size; }
    auto width() const noexcept { return this->_size.x(); }
    auto height() const noexcept { return this->_size.y(); }
    auto stride() const noexcept { return this->_stride; }
    auto sizeInBytes() const noexcept { return this->_data.count * sizeof(this->_data.data.get()[0]); };
    auto data() noexcept { return this->_data.data.get(); }
    const auto data() const noexcept { return this->_data.data.get(); }
    template <typename T = uint8_t>
    auto value(int32_t x, int32_t y) const
    {
        return static_cast<T>(this->_data[y * this->_stride + x]);
    }
    template <typename T, typename U>
    auto value(const Point2d<U>& pt) const
    {
        return this->value<T>(pt.x(), pt.y());
    }
    void setValue(int32_t x, int32_t y, uint8_t value)
    {
        this->_data[y * this->_stride + x] = value;
    }
    auto copy() const
    {
        ImagePlane result(this->_size, this->_stride);
        result._data = this->_data.clone();
        return result;
    }

private:
    AlignedBuffer<uint8_t, 32> _data;
    uint32_t _stride = 0;
    Size32u _size;
};

template <uint8_t numberOfPlanes>
class Image2 {
public:
    using Planes = std::array<ImagePlane, numberOfPlanes>;

    Image2() {}
    Image2(Planes&& p)
        : _planes(std::move(p))
    {
    }

    constexpr auto planeCount() const noexcept { return this->_planes.size(); }
    Planes& planes() noexcept { return this->_planes; }
    const Planes& planes() const noexcept { return this->_planes; }
    ImagePlane& plane(size_t index) noexcept { return this->_planes[index]; }
    const ImagePlane& plane(size_t index) const noexcept { return this->_planes[index]; }

private:
    Planes _planes;
};
}
