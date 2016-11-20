#include "CudaEncoderBackend.h"
#include <iostream>
#include "transform.h"
#include "encode/datatypes.h"

#ifdef FRAC_WITH_CUDA

using namespace Frac;

class GpuSamplerBilinear {
public:
	CUDA_CALLABLE GpuSamplerBilinear(const uint8_t* buffer, const cuda_partition_item_t& item, const uint32_t stride)
		: _buffer(buffer)
		, _item(item)
		, _stride(stride)
	{}
	CUDA_CALLABLE ~GpuSamplerBilinear() {}
	CUDA_CALLABLE uint8_t operator()(const uint32_t x, const uint32_t y) const {
		const int valB_0 = static_cast<int>(this->_buffer[x + this->_item.x + (y + this->_item.y) * this->_stride]);
		const int valB_1 = static_cast<int>(this->_buffer[x + this->_item.x + 1 + (y + this->_item.y) * this->_stride]);
		const int valB_2 = static_cast<int>(this->_buffer[x + this->_item.x + (y + this->_item.y + 1) * this->_stride]);
		const int valB_3 = static_cast<int>(this->_buffer[x + this->_item.x + 1 + (y + this->_item.y + 1) * this->_stride]);
		return static_cast<uint8_t>((valB_0 + valB_1 + valB_2 + valB_3) / 4);
	}
	CUDA_CALLABLE uint8_t operator() (uint32_t x, uint32_t y, const Transform& t) const {
		if (x == _item.width - 1)
			--x;
		if (y == _item.height - 1)
			--y;
		cuda_size_t tl, tr, bl, br;
		t.map(&tl.x, &tl.y, x, y, _item.width, _item.height);
		t.map(&tr.x, &tr.y, x + 1, y, _item.width, _item.height);
		t.map(&bl.x, &bl.y, x, y + 1, _item.width, _item.height);
		t.map(&br.x, &br.y, x + 1, y + 1, _item.width, _item.height);
		const int total = (int)this->_buffer[tl.x + this->_item.x + (tl.y + this->_item.y) * _stride]
			+ (int)this->_buffer[tr.x + this->_item.x + (tr.y + this->_item.y) * _stride]
			+ (int)this->_buffer[bl.x + this->_item.x + (bl.y + this->_item.y) * _stride]
			+ (int)this->_buffer[br.x + this->_item.x + (br.y + this->_item.y) * _stride];
		return static_cast<uint8_t>(total / 4);
	}
private:
	const uint8_t* _buffer;
	const cuda_partition_item_t& _item;
	const uint32_t _stride;
};

__global__ static void encode_kernel(const cuda_launch_params_t params,
	const cuda_partition_item_t targetItem,
	cuda_thread_result_t* result)
{
	const int idInBlock = threadIdx.x + threadIdx.y * blockDim.x;
	const int countInBlock = blockDim.x * blockDim.y;
	const int countPreviousBlocks = blockIdx.x + blockIdx.y * gridDim.x;
	const int myId = idInBlock + countInBlock * countPreviousBlocks;
	if (myId >= params.partitionSize)
		return;
	const cuda_partition_item_t sourceItem = params.partition[myId];

	Transform t(Transform::Id);
	result[myId].distance = 9999.0;
	result[myId].index = myId;
	do {
		const double N = (double)(targetItem.height) * targetItem.width;
		double sumA = 0.0, sumA2 = 0.0, sumB = 0.0, sumAB = 0.0;
		double dist = 0.0;
		const GpuSamplerBilinear sampler(params.gpuBuffer, sourceItem, params.stride);
		for (uint32_t y = 0; y < targetItem.height; ++y) {
			for (uint32_t x = 0; x < targetItem.width; ++x) {
				const auto targetX = x + targetItem.x;
				const auto targetY = y + targetItem.y;
				const auto srcX = (x * sourceItem.height) / targetItem.height;
				const auto srcY = (y * sourceItem.width) / targetItem.width;
				const double valA = static_cast<double>(params.gpuBuffer[targetX + targetY * params.stride]);
				const int valB = static_cast<int>(sampler(srcX, srcY, t));
				sumA += valA;
				sumB += valB;
				sumA2 += valA * valA;
				sumAB += valA * valB;
				dist += (valA - valB) * (valA - valB);
			}
		}
		dist = sqrt(dist / N);
		if (dist < result[myId].distance) {
			const double tmp = (N * sumA2 - (sumA - 1) * sumA);
			const double s = (fabs(tmp) < 0.00001) ? 0.0 : (N * sumAB - sumA * sumB) / tmp;
			const double o = (sumB - s * sumA) / N;
			result[myId].distance = dist;
			result[myId].contrast = s;
			result[myId].brightness = o;
			result[myId].transform = static_cast<int>(t.type());
		}
		if (dist <= params.params.rmsThreshold)
			break;
	} while (t.next() != Transform::Id);
}

CudaEncodeKernel::CudaEncodeKernel(size_t width, size_t height, size_t stride, const encode_parameters_t& params, uint8_t* buffer, const cuda_partition_item_t* partition, const size_t partitionSize)
{
	_kernelParams.width = width;
	_kernelParams.height = height;
	_kernelParams.stride = stride;
	_kernelParams.params = params;
	_kernelParams.gpuBuffer = buffer;
	_kernelParams.partition = partition;
	_kernelParams.partitionSize = partitionSize;
	int minGridSize = 0;
	int blockSize = 0;
	CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (CUfunction)encode_kernel, 0, 0));
	_blockSize.x = blockSize / 32;
	_blockSize.y = 32;
	blockSize = 4;
	int totalThreads = 0;
	do {
		_gridSize.x = blockSize;
		_gridSize.y = blockSize;
		totalThreads = _blockSize.x * _blockSize.y * _gridSize.x * _gridSize.y;
		blockSize *= 2;
	} while (totalThreads < partitionSize);
	std::cout << "configuration: " << _gridSize.x * _gridSize.y << " blocks / " << _blockSize.x << 'x' << _blockSize.y << " threads; " << totalThreads << " total.\n";
	CUDA_CALL(cudaMallocHost(&_kernelResult, partitionSize * sizeof(cuda_thread_result_t)));
}

cuda_thread_result_t CudaEncodeKernel::launch(const cuda_partition_item_t& targetItem) {
	const dim3 blockSize(_blockSize.x, _blockSize.y);
	const dim3 gridSize(_gridSize.x, _gridSize.y);
	memset(_kernelResult, 0, sizeof(_kernelResult) * _kernelParams.partitionSize);
	encode_kernel<<< blockSize, gridSize >>>(this->_kernelParams, targetItem, this->_kernelResult);
	CUDA_CALL(cudaDeviceSynchronize());
	size_t index = 0;
	double minDist = _kernelResult[0].distance;
	for (size_t i = 1; i < _kernelParams.partitionSize; ++i) {
		if (minDist > _kernelResult[i].distance) {
			minDist = _kernelResult[i].distance;
			index = i;
			if (minDist < _kernelParams.params.rmsThreshold)
				break;
		}
	}
	return _kernelResult[index];
}

#endif
