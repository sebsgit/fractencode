#include "CudaEncodingEngine.h"
#include "CudaEncoderBackend.h"

#ifdef FRAC_WITH_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

using namespace Frac;

class CudaEncodingEngine::CudaEncoderBackend {
public:
	CudaEncoderBackend(const encode_parameters_t& params, const Image& image, const PartitionPtr& sourceGrid)
		: _parameters(params)
	{
		int device = -1;
		cudaDeviceProp props;
		CUDA_CALL(cudaGetDevice(&device));
		CUDA_CALL(cudaGetDeviceProperties(&props, device));
		std::cout << "cuda backend on " << props.name << '\n';
		const auto imageBuffer = image.data();
		CUDA_CALL(cudaMalloc(&_gpuBuffer, imageBuffer->size() * sizeof(Image::Pixel)));
		CUDA_CALL(cudaMemcpy(_gpuBuffer, imageBuffer->get(), imageBuffer->size(), cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMallocHost(&_partition, sourceGrid->size() * sizeof(cuda_partition_item_t)));
		size_t i = 0;
		for (auto item : *sourceGrid) {
			this->_partition[i] = cuda_partition_item_t(item->x(), item->y(), item->width(), item->height());
			++i;
		}
		this->_kernel.reset(new CudaEncodeKernel(image.width(), image.height(), image.stride(), this->_parameters, this->_gpuBuffer, this->_partition, sourceGrid->size()));
	}
	~CudaEncoderBackend() {
		if (_gpuBuffer)
			CUDA_CALL(cudaFree(this->_gpuBuffer));
		if (_partition)
			CUDA_CALL(cudaFreeHost(this->_partition));
	}
	encode_item_t encode(const PartitionItemPtr& targetItem) {
		const cuda_partition_item_t targetRect(targetItem->x(), targetItem->y(), targetItem->width(), targetItem->height());
		const cuda_thread_result_t kernelResult = this->_kernel->launch(targetRect);
		encode_item_t result;
		result.match.score.distance = kernelResult.distance;
		result.match.score.brightness = kernelResult.brightness;
		result.match.score.contrast = kernelResult.contrast;
		result.match.score.transform = static_cast<Transform::Type>(kernelResult.transform);
		result.match.sourceItemSize = Size32u(this->_partition[kernelResult.index].width, this->_partition[kernelResult.index].height);
		result.match.x = this->_partition[kernelResult.index].x;
		result.match.y = this->_partition[kernelResult.index].y;
		//std::cout << "CUDA DEBUG ! " << kernelResult.index << ' ' << this->_partition[kernelResult.index].x << ' ' << this->_partition[kernelResult.index].y << ' ' << this->_partition[kernelResult.index].width << ' ' << this->_partition[kernelResult.index].height << '\n';
		result.x = targetItem->x();
		result.y = targetItem->y();
		result.w = targetItem->width();
		result.h = targetItem->height();
		return result;
	}
private:
	Image::Pixel* _gpuBuffer = nullptr;
	cuda_partition_item_t* _partition = nullptr;
	const encode_parameters_t _parameters;
	std::unique_ptr<CudaEncodeKernel> _kernel;
};

CudaEncodingEngine::~CudaEncodingEngine() {
	
}

void CudaEncodingEngine::init() {
	this->_backend.reset(new CudaEncoderBackend(this->_parameters, this->_image, this->_source));
}

void CudaEncodingEngine::finalize() {
	this->_backend.reset();
}

encode_item_t CudaEncodingEngine::encode_impl(const PartitionItemPtr& target) const {
	return this->_backend->encode(target);
}

#endif // FRAC_WITH_CUDA
