#ifndef FRAC_BUFFER_HPP
#define FRAC_BUFFER_HPP

#include <memory>
#include <cassert>
#include <functional>
#include <inttypes.h>
#include <cstring>

namespace Frac {

template <typename T, typename U>
const T convert(const U u) {
	return static_cast<T>(u);
}

}

#endif // FRAC_BUFFER_HPP
