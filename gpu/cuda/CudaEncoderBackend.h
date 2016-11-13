#pragma once

#include "CudaConf.h"

#ifdef FRAC_WITH_CUDA

#include <cuda_runtime.h>
#include <inttypes.h>
#include "encode/encode_parameters.h"

namespace Frac {
	typedef struct _cuda_partition_item_t {
		uint32_t x, y, width, height;
		_cuda_partition_item_t(uint32_t _x, uint32_t _y, uint32_t w, uint32_t h)
			:x(_x), y(_y), width(w), height(h)
		{}
	} cuda_partition_item_t;

	typedef struct {
		uint32_t x, y;
	} cuda_size_t;

	typedef struct {
		size_t width;
		size_t height;
		size_t stride;
		encode_parameters_t params;
		uint8_t* gpuBuffer;
		const cuda_partition_item_t* partition;
		size_t partitionSize;
	} cuda_launch_params_t;

	typedef struct {
		double distance = 999.0;
		double contrast = 1.0;
		double brightness = 0.0;
		size_t index = 0;
		int transform = 0;
	} cuda_thread_result_t;

	class CudaEncodeKernel {
	public:
		CudaEncodeKernel(size_t width, size_t height, size_t stride, const encode_parameters_t& params, uint8_t* buffer, const cuda_partition_item_t* partition, const size_t partitionSize)
		{
			_kernelParams.width = width;
			_kernelParams.height = height;
			_kernelParams.stride = stride;
			_kernelParams.params = params;
			_kernelParams.gpuBuffer = buffer;
			_kernelParams.partition = partition;
			_kernelParams.partitionSize = partitionSize;
			_blockSize.x = 16;
			_blockSize.y = 16;
			_gridSize.x = 16;
			_gridSize.y = 16;
			cudaMallocHost(&_kernelResult, partitionSize * sizeof(cuda_thread_result_t));
		}
		~CudaEncodeKernel() {
			cudaFreeHost(_kernelResult);
		}
		cuda_thread_result_t launch(const cuda_partition_item_t& targetItem);
	private:
		cuda_launch_params_t _kernelParams;
		cuda_size_t _blockSize, _gridSize;
		cuda_thread_result_t* _kernelResult;
	};
}

#endif
