#include "CudaEncoderBackend.h"
#include <iostream>

#ifdef FRAC_WITH_CUDA

using namespace Frac;

/*
transform_score_t match_default(const PartitionItemPtr& target, const PartitionItemPtr& source) const {
		transform_score_t result;
		Transform t(Transform::Id);
		const SamplerBilinear samplerB(source->image());
		do {
			transform_score_t candidate;
			candidate.distance = _metric.distance(target->image(), source->presampled(), t);
			candidate.transform = t.type();
			if (candidate.distance <= result.distance) {
				const double N = (double)(target->image().width()) * target->image().height();
				double sumA = ImageStatistics::sum(target->image()), sumA2 = 0.0, sumB = 0.0, sumAB = 0.0;
				for (uint32_t y = 0 ; y<target->image().height() ; ++y) {
					for (uint32_t x = 0 ; x<target->image().width() ; ++x) {
						const auto srcY = (y * source->image().height()) / target->image().height();
						const auto srcX = (x * source->image().width()) / target->image().width();
						const double valA = convert<double>(target->image().data()->get()[x + y * target->image().stride()]);
						const double valB = convert<double>(samplerB(srcX, srcY, t, source->image().size()));
						sumB += valB;
						sumA2 += valA * valA;
						sumAB += valA * valB;
					}
				}
				const double tmp = (N * sumA2 - (sumA - 1) * sumA);
				const double s = this->truncateSMax( fabs(tmp) < 0.00001 ? 0.0 : (N * sumAB - sumA * sumB) / tmp );
				const double o = (sumB - s * sumA) / N;
				candidate.contrast = s;
				candidate.brightness = o;
				result = candidate;
			}
			if (this->checkDistance(result.distance))
				break;
		} while (t.next() != Transform::Id);
		return result;
	}
*/

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
	const double N = (double)(targetItem.height) * targetItem.width;
	double sumA = 0.0, sumA2 = 0.0, sumB = 0.0, sumAB = 0.0;
	double dist = 0.0;
	for (uint32_t y = 0; y<targetItem.height; ++y) {
		for (uint32_t x = 0; x<targetItem.width; ++x) {
			const auto targetX = x + targetItem.x;
			const auto targetY = y + targetItem.y;
			const auto srcX = (x * sourceItem.height) / targetItem.height;
			const auto srcY = (y * sourceItem.width) / targetItem.width;
			const double valA = static_cast<double>(params.gpuBuffer[targetX + targetY * params.stride]);
			const int valB_0 = static_cast<int>(params.gpuBuffer[srcX + sourceItem.x + (srcY + sourceItem.y) * params.stride]);
			const int valB_1 = static_cast<int>(params.gpuBuffer[srcX + sourceItem.x + 1 + (srcY + sourceItem.y) * params.stride]);
			const int valB_2 = static_cast<int>(params.gpuBuffer[srcX + sourceItem.x + (srcY + sourceItem.y + 1) * params.stride]);
			const int valB_3 = static_cast<int>(params.gpuBuffer[srcX + sourceItem.x + 1 + (srcY + sourceItem.y + 1) * params.stride]);
			const int valB = (valB_0 + valB_1 + valB_2 + valB_3) / 4;
			sumA += valA;
			sumB += valB;
			sumA2 += valA * valA;
			sumAB += valA * valB;
			dist += (valA - valB) * (valA - valB);
		}
	}
	dist = sqrt(dist / N);
	const double tmp = (N * sumA2 - (sumA - 1) * sumA);
	const double s = (fabs(tmp) < 0.00001) ? 0.0 : (N * sumAB - sumA * sumB) / tmp;
	const double o = (sumB - s * sumA) / N;
	result[myId].index = myId;
	result[myId].distance = dist;
	result[myId].contrast = s;
	result[myId].brightness = o;
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
