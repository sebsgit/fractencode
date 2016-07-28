#ifndef POINT2D_HPP
#define POINT2D_HPP

#include <inttypes.h>

namespace Frac {
    template <typename T>
    class Point2d {
    public:
        Point2d() : Point2d(T(), T()) {}
        Point2d(const T x, const T y)
            :_x(x), _y(y)
        {

        }
        const T x() const noexcept {
            return _x;
        }
        const T y() const noexcept {
            return _y;
        }
    private:
        T _x, _y;
    };

    using Point2du = Point2d<uint32_t>;
}

#endif // POINT2D_HPP
