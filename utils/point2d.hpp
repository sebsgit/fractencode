#ifndef POINT2D_HPP
#define POINT2D_HPP

#include <inttypes.h>
#include <ostream>

namespace Frac {
	template <typename T>
	class Point2d {
	public:
        struct hash {
            constexpr auto operator() (const Point2d& p) const noexcept {
                return p._x ^ p._y;
            }
        };

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
		T& x() { return _x; }
		T& y() { return _y; }

        friend std::ostream& operator << (std::ostream& out, const Point2d<T>& p) {
            out << p.x() << ',' << p.y() << ' ';
            return out;
        }

        bool operator==(const Point2d& other) const noexcept {
            return other._x == this->_x && other._y == this->_y;
        }

        friend Point2d operator+ (const Point2d& p0, const Point2d& p1) noexcept
        {
            return Point2d{ p0._x + p1._x, p0._y + p1._y };
        }
	private:
		T _x, _y;
	};

	using Point2du = Point2d<uint32_t>;
}

#endif // POINT2D_HPP
