#ifndef POINT2D_HPP
#define POINT2D_HPP

#include <inttypes.h>
#include <ostream>

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
		T& x() { return _x; }
		T& y() { return _y; }
	private:
		T _x, _y;
	};

	using Point2du = Point2d<uint32_t>;
}

extern std::ostream& operator << (std::ostream& out, const Frac::Point2du& p);

#endif // POINT2D_HPP
