#pragma once

#include "utils/size.hpp"
#include "utils/point2d.hpp"
#include "utils/attributes.hpp"

#include <vector>
#include <array>

namespace Frac2 {
    using namespace Frac;

    class ImagePlane {
    public:
        FRAC_NO_COPY(ImagePlane);

        ImagePlane(const Size32u& size, uint32_t stride)
            : _data(size.y() * stride, 0)
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
            : _data(std::move(initialData))
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
        auto data() noexcept { return this->_data.data(); }
        const auto data() const noexcept { return this->_data.data(); }
        template <typename T = uint8_t>
        auto value(int32_t x, int32_t y) const {
            return static_cast<T>(this->_data[y * this->_stride + x]);
        }
        template <typename T, typename U>
        auto value(const Point2d<U>& pt) const {
            return this->value<T>(pt.x(), pt.y());
        }
        auto copy() const {
            ImagePlane result(this->_size, this->_stride);
            result._data = this->_data;
            return result;
        }
    private:
        std::vector<uint8_t> _data;
        uint32_t _stride = 0;
        Size32u _size;
    };

    template <uint8_t numberOfPlanes>
    class Image2 {
    public:
        using Planes = std::array<ImagePlane, numberOfPlanes>;

        Image2() {}
        Image2(Planes&& p) : _planes(std::move(p)) {}
        
        constexpr auto planeCount() const noexcept { return this->_planes.size(); }
        Planes& planes() noexcept { return this->_planes; }
        const Planes& planes() const noexcept { return this->_planes; }
        ImagePlane& plane(size_t index) noexcept { return this->_planes[index]; }
        const ImagePlane& plane(size_t index) const noexcept { return this->_planes[index]; }

    private:
        Planes _planes;
    };
}
