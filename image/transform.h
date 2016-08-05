#ifndef FRAC_TRANSFORM_H_
#define FRAC_TRANSFORM_H_

#include "image/image.h"
#include "utils/size.hpp"
#include "utils/point2d.hpp"
#include "image/sampler.h"

namespace Frac {
    class Transform {
    public:
        enum Type {
            Id,
            Rotate_90,
            Rotate_180,
            Rotate_270,
            Flip,
            Flip_Rotate_90,
            Flip_Rotate_180,
            Flip_Rotate_270
        };
        enum Interpolation {
            NearestNeighbor,
            Bilinear
        };

        Transform(Type t = Id) noexcept
            : _type(t)
        {

        }
        void setType(const Type t) noexcept {
            this->_type = t;
        }
        Type type() const noexcept {
            return _type;
        }
        Type next() {
            if (_type == Flip_Rotate_270) {
                _type = Id;
            } else {
                _type = static_cast<Type>((int)(_type + 1));
            }
            return _type;
        }
        Transform inverse() const {
            switch (_type) {
            case Rotate_90:
                return Transform(Rotate_270);
            case Rotate_270:
                return Transform(Rotate_90);
            case Flip_Rotate_90:
                return Transform(Flip_Rotate_270);
            case Flip_Rotate_270:
                return Transform(Flip_Rotate_90);
            default:
                return *this;
            }
        }
        Image resize(const Image& source, const Size32u& targetSize, Interpolation t = NearestNeighbor) const {
            switch(t) {
            case Bilinear:
                return this->_resize_b(source, targetSize);
            default:
                return this->_resize_nn(source, targetSize);
            }
        }
        Image map(const Image& source) const {
            if (this->_type == Id)
                return source;
            const Size32u targetSize = this->map(source.size());
            AbstractBufferPtr<Image::Pixel> buffer = Buffer<Image::Pixel>::alloc(targetSize.x() * targetSize.y());
            const auto* sourcePtr = source.data()->get();
            auto* targetPtr = buffer->get();
            Image result(buffer, targetSize.x(), targetSize.y(), targetSize.x());
            result.map([&](uint32_t x, uint32_t y) {
                const Point2du p = this->map(x, y, targetSize);
                targetPtr[x + y * targetSize.x()] = sourcePtr[p.x() + p.y() * source.stride()];
            });
            return result;
        }
        Size32u map(const Size32u& s) const noexcept {
            switch(_type) {
            case Rotate_90:
            case Rotate_270:
            case Flip_Rotate_90:
            case Flip_Rotate_270:
                return Size32u(s.y(), s.x());
            default:
                return s;
            }
        }
        Point2du map(uint32_t x, uint32_t y, const Size32u& s) const noexcept {
            switch(_type) {
            case Rotate_90:
                return Point2du(y, s.x() - x);
            case Rotate_180:
                return Point2du(s.x() - x, s.y() - y);
            case Rotate_270:
                return Point2du(s.y() - y, x);
            case Flip:
                return Point2du(x, s.y() - y);
            case Flip_Rotate_90:
                return Point2du(y, x);
            case Flip_Rotate_180:
                return Point2du(s.x() - x, y);
            case Flip_Rotate_270:
                return Point2du(s.y() - y, s.x() - x);
            default:
                return Point2du(x, y);
            }
        }
        void copy(const Image& source, Image& target, const double contrast = 1.0, const double brightness = 0.0) const;
    private:
        Image _resize_nn(const Image& source, const Size32u& targetSize) const;
        Image _resize_b(const Image& source, const Size32u& targetSize) const;
        template <typename Sampler>
        Image _resize_impl(const Image& source, const Size32u& targetSize) const {
            if (source.size() != targetSize || this->_type != Id) {
                const Sampler samplerSource(source);
                AbstractBufferPtr<Image::Pixel> buffer = Buffer<Image::Pixel>::alloc(targetSize.x() * targetSize.y());
                auto* targetPtr = buffer->get();
                for (uint32_t y = 0 ; y<targetSize.y() ; ++y) {
                    for (uint32_t x = 0 ; x<targetSize.x() ; ++x) {
                        const uint32_t srcY = (y * source.height()) / targetSize.y();
                        const uint32_t srcX = (x * source.width()) / targetSize.x();
                        const auto p = this->map(srcX, srcY, source.size());
                        targetPtr[x + y * targetSize.x()] = samplerSource(p.x(), p.y());
                    }
                }
                return Image(buffer, targetSize.x(), targetSize.y(), targetSize.x());
            } else {
                return source;
            }
        }
    private:
        Type _type = Id;
    };
}

#endif
