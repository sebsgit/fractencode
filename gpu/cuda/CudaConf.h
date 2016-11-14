#pragma once

#define FRAC_WITH_CUDA

#ifdef FRAC_WITH_CUDA

#include <cuda.h>
#include <iostream>

#define CUDA_CALL(what) { auto result = what ; if (result != cudaError::cudaSuccess) { std::cout << "error while calling " #what ": " << cudaGetErrorName(result) << '\n'; exit(0); } }
#endif
