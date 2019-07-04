#pragma once

#include "utils/Config.h"

#ifdef FRAC_WITH_CUDA

#include <cuda.h>
#include <iostream>

#define CUDA_CALL(what) { auto _tmp_resultvar = what ; if (_tmp_resultvar != cudaError::cudaSuccess) { std::cout << "error while calling " #what ": " << cudaGetErrorName(_tmp_resultvar) << '\n'; exit(0); } }

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#endif

#endif

#ifndef CUDA_CALLABLE
#define CUDA_CALLABLE
#endif
