#ifndef FRAC_BUFFER_HPP
#define FRAC_BUFFER_HPP

#include <memory>
#include <cassert>
#include <functional>
#include <inttypes.h>
#include <cstring>

namespace Frac {

static double __u2d_lookup[] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
    32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,
    64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,
    96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,
    128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,
    160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,
    192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,
    224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255
};

template <typename T, typename U>
T convert(const U u) {
    return static_cast<T>(u);
}

template <>
double convert(const uint8_t u) {
    return __u2d_lookup[u];
}

template <>
float convert(const uint8_t u) {
    return __u2d_lookup[u];
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

}

#endif // FRAC_BUFFER_HPP
