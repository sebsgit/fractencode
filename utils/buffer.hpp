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

template <typename T>
class NewAllocator {
public:
	static T* acquire(const size_t n) {
		return new T[n];
	}
	static void release(T* ptr) {
		delete[] ptr;
	}
	static void memset(T* ptr, const int value, const size_t size) {
		::memset(ptr, value, size);
	}
	static void memcpy(T* dest, const T* src, const size_t size) {
		std::copy(src, src + size, dest);
	}
};

template <typename T, template <typename> class Allocator = NewAllocator>
class AbstractBuffer {
public:
	virtual ~AbstractBuffer() {}
	virtual const T* get() const = 0;
	virtual size_t size() const = 0;
	virtual T* get() = 0;
	virtual std::shared_ptr<AbstractBuffer<T>> clone() const = 0;
	void memset(const uint8_t value, size_t size = 0) noexcept {
		if (size == 0)
			size = this->size();
		Allocator<T>::memset(this->get(), value, size);
	}
};

template <typename T, template <typename> class Allocator = NewAllocator>
using AbstractBufferPtr = std::shared_ptr<AbstractBuffer<T, Allocator>>;

template <typename T, template <typename> class Allocator = NewAllocator>
class Buffer : public AbstractBuffer<T, Allocator> {
public:
	Buffer(T* data, const size_t size, std::function<void(T*)> deleter = Allocator<T>::release)
		:_data(data, deleter)
		,_size(size)
	{
		assert(data);
	}
	virtual ~Buffer() {}
	virtual const T* get() const override {
		return _data.get();
	}
	virtual T* get() override {
		return _data.get();
	}
	virtual size_t size() const override {
		return _size;
	}
	virtual AbstractBufferPtr<T, Allocator> clone() const override {
		auto result = alloc(this->size());
		Allocator<T>::memcpy(result->get(), this->get(), this->size());
		return result;
	}
	static AbstractBufferPtr<T, Allocator> alloc(const uint64_t size) {
		return AbstractBufferPtr<T, Allocator>(new Buffer<T, Allocator>(Allocator<T>::acquire(size), size));
	}
private:
	std::shared_ptr<T> _data;
	size_t _size;
};

template <typename T, template <typename> class Allocator = NewAllocator>
class BufferSlice : public AbstractBuffer<T, Allocator> {
public:
	BufferSlice(AbstractBufferPtr<T, Allocator> source, const uint64_t offset, const size_t size)
		:_source(source)
		,_offset(offset)
		,_size(size)
	{

	}
	const T* get() const override {
		return _source->get() + _offset;
	}
	T* get() override {
		return _source->get() + _offset;
	}
	size_t size() const override {
		return _size;
	}
	virtual AbstractBufferPtr<T, Allocator> clone() const override {
		auto result = Buffer<T, Allocator>::alloc(this->size());
		Allocator<T>::memcpy(result->get(), this->get(), this->size());
		return result;
	}
	static AbstractBufferPtr<T, Allocator> slice(AbstractBufferPtr<T, Allocator> source, uint64_t offset, size_t size) {
		return AbstractBufferPtr<T, Allocator>(new BufferSlice<T, Allocator>(source, offset, size));
	}
private:
	AbstractBufferPtr<T, Allocator> _source;
	uint64_t _offset;
	size_t _size;
};

template <typename U, typename T, template <typename> class AllocatorSource = NewAllocator, template <typename> class AllocatorResult = NewAllocator>
AbstractBufferPtr<U, AllocatorResult> convert(AbstractBufferPtr<T, AllocatorSource> source) {
	auto result = Buffer<U, AllocatorResult>::alloc(source->size());
	for (size_t i=0 ; i<source->size() ; ++i)
		result->get()[i] = convert<T, U>(source->get()[i]);
	return result;
}

}

#endif // FRAC_BUFFER_HPP
