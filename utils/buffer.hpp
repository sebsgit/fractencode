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
        ::memset(this->get(), value, size);
    }
};

template <typename T>
using AbstractBufferPtr = std::shared_ptr<AbstractBuffer<T>>;

template <typename T>
class Buffer : public AbstractBuffer<T> {
public:
    Buffer(T* data, const size_t size, std::function<void(T*)> deleter = [](T* d){ free(d); })
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
    virtual AbstractBufferPtr<T> clone() const override {
        auto result = alloc(this->size());
        std::copy(this->get(), this->get() + this->size() , result->get());
        return result;
    }
    static AbstractBufferPtr<T> alloc(const uint64_t size) {
        return AbstractBufferPtr<T>(new Buffer<T>(new T[size], size, [](T* d){ delete [] d; }));
    }
private:
    std::shared_ptr<T> _data;
    size_t _size;
};

template <typename T>
class BufferSlice : public AbstractBuffer<T> {
public:
    BufferSlice(AbstractBufferPtr<T> source, const uint64_t offset, const size_t size)
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
    virtual AbstractBufferPtr<T> clone() const override {
        auto result = Buffer<T>::alloc(this->size());
        std::copy(this->get(), this->get() + this->size() , result->get());
        return result;
    }
    static AbstractBufferPtr<T> slice(AbstractBufferPtr<T> source, uint64_t offset, size_t size) {
        return AbstractBufferPtr<T>(new BufferSlice<T>(source, offset, size));
    }
private:
    AbstractBufferPtr<T> _source;
    uint64_t _offset;
    size_t _size;
};

template <typename U, typename T>
AbstractBufferPtr<U> convert(AbstractBufferPtr<T> source) {
    auto result = Buffer<U>::alloc(source->size());
    for (size_t i=0 ; i<source->size() ; ++i)
        result->get()[i] = convert<T, U>(source->get()[i]);
    return result;
}

}

#endif // FRAC_BUFFER_HPP
