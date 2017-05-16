#pragma once

#include <algorithm>
#include <type_traits>

namespace Frac {
template <typename T>
class Quantizer {
	static_assert(std::is_arithmetic<T>::value, "cannot quantize non-arithmetic data");
public:
	using Int = uint64_t;

	explicit constexpr Quantizer(T minValue, T maxValue, int numberOfBits)
		:_min(minValue)
		, _max(maxValue)
		, _bits(numberOfBits)
		, _step(std::abs(maxValue - minValue) / (1 << numberOfBits))
		, _maxQuantized((1 << numberOfBits) - 1)
	{
		assert(maxValue > minValue);
		assert(numberOfBits > 1);
		assert(numberOfBits <= 64);
		assert(_step > 0.0);
	}
	constexpr Int quantized(T value) const
	{
		assert(value <= this->_max);
		assert(value >= this->_min);
		return std::min(this->_maxQuantized, static_cast<Int>(std::floor((value - this->_min) / (this->_step))));
	}
	constexpr T value(Int quant) const
	{
		assert(quant >= 0);
		assert(quant < std::pow(2, this->_bits));
		return quant * this->_step + this->_min + this->_step / 2;
	}
private:
	const T _min;
	const T _max;
	const int _bits;
	const T _step;
	const Int _maxQuantized;
};

using Quantizerd = Quantizer<double>;
}
