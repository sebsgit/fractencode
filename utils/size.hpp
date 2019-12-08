#ifndef SIZE_HPP
#define SIZE_HPP

#include <inttypes.h>
#include <iostream>

namespace Frac {
	template <typename T>
	class Size {
	public:
        constexpr Size() noexcept : _x(T()), _y(T()) {

		}
        constexpr Size(T x, T y) noexcept
			:_x(x), _y(y)
		{}
		
		constexpr Size(const Size& other) noexcept = default;
		constexpr Size& operator= (const Size& other) noexcept = default;
        constexpr Size(Size&& other) noexcept = default;
		constexpr Size& operator= (Size&& other) noexcept = default;
        
        T x() const noexcept {
			return _x;
		}
        T y() const noexcept {
			return _y;
		}
		void setX(const T& x) noexcept {
			_x = x;
		}
		void setY(const T& y) noexcept {
			_y = y;
		}
		bool operator==(const Size& other) const {
			return _x == other._x && _y == other._y;
		}
		bool operator != (const Size& other) const {
			return !(*this == other);
		}
		Size operator / (const T& value) const {
			return Size(_x / value, _y / value);
		}
		Size operator * (const T& value) const {
			return Size(_x * value, _y * value);
		}
		bool isAligned(const T x, const T y) const {
			return _x % x == 0 && _y % y == 0;
		}
		Size align(const T x, const T y) const {
			const T rx = _x % x;
			const T ry = _y % y;
			return Size(_x + (rx ? (x - rx) : 0), _y + (ry ? (y - ry) : 0));
		}
		const T area() const {
			return _x * _y;
		}

        friend std::ostream& operator << (std::ostream& out, const Size& s) {
            return out << '{' << s.x() << 'x' << s.y() << '}';
        }
	private:
		T _x, _y;
	};
	using Size32u = Size<uint32_t>;
}

#endif // SIZE_HPP
