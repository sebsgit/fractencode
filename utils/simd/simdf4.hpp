#pragma once

#if !defined(__ANDROID__)
#include "x86/simdf4_sse.hpp"
#else
#include "neon/simdf4_neon.hpp"
#endif
