#pragma once
#include "CudaConf.h"
#ifdef FRAC_WITH_CUDA

#include <cuda_runtime.h>

namespace Frac {

class PageLockedBufferAllocator {
public:
	static constexpr bool directlyAddressable = true;
	static void* alloc(size_t size) {
		void* result = nullptr;
		CUDA_CALL(cudaMallocHost(&result, size));
		return result;
	}
	static void free(void* buffer) {
		CUDA_CALL(cudaFreeHost(buffer));
	}
	static void copyToDevice(void* destination, const void* source, size_t size) {
		memcpy(destination, source, size);
	}
	static void copyToHost(void* destination, const void* source, size_t size) {
		memcpy(destination, source, size);
	}
};

class DeviceBufferAllocator {
public:
	static constexpr bool directlyAddressable = false;
	static void* alloc(size_t size) {
		void* result = nullptr;
		CUDA_CALL(cudaMalloc(&result, size));
		return result;
	}
	static void free(void* buffer) {
		CUDA_CALL(cudaFree(buffer));
	}
	static void copyToDevice(void* destination, const void* source, size_t size) {
		CUDA_CALL(cudaMemcpy(destination, source, size, cudaMemcpyHostToDevice));
	}
	static void copyToHost(void* destination, const void* source, size_t size) {
		CUDA_CALL(cudaMemcpy(destination, source, size, cudaMemcpyDeviceToHost));
	}
};

template <typename T, typename Alloc = DeviceBufferAllocator>
class CudaPtr {
public:
	CudaPtr() : _buffer(nullptr) {

	}
	explicit CudaPtr(const T* hostMemory, size_t count = 1) : CudaPtr(count) {
		Alloc::copyToDevice(this->_buffer, hostMemory, count * sizeof(T));
	}
	explicit CudaPtr(size_t count) {
		assert(count >= 1);
		this->_buffer = static_cast<T*>(Alloc::alloc(count * sizeof(T)));
	}
	~CudaPtr() {
		if (this->_buffer)
			Alloc::free(this->_buffer);
	}
	CudaPtr(const CudaPtr& other) = delete;
	CudaPtr(CudaPtr&& other) noexcept {
		this->_buffer = other._buffer;
		other._buffer = nullptr;
	}
	CudaPtr& operator=(const CudaPtr& other) = delete;
	CudaPtr& operator= (CudaPtr&& other) noexcept {
		this->_buffer = other._buffer;
		other._buffer = nullptr;
		return *this;
	}
	void realloc(size_t count) {
		if (this->_buffer)
			Alloc::free(this->_buffer);
		this->_buffer = static_cast<T*>(Alloc::alloc(count * sizeof(T)));
	}
	const T& operator[](size_t index) const noexcept {
		static_assert(Alloc::directlyAddressable, "operator[] available only for directly-addressable memory");
		return this->_buffer[index];
	}
	T& operator[](size_t index) noexcept {
		static_assert(Alloc::directlyAddressable, "operator[] available only for directly-addressable memory");
		return this->_buffer[index];
	}
	const T* get() const noexcept {
		return this->_buffer;
	}
	T* get() noexcept {
		return this->_buffer;
	}
private:
	T* _buffer;
};

}

#endif
