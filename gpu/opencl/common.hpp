#ifndef CLCOMMON_HPP
#define CLCOMMON_HPP

#include "image/partition2.hpp"

namespace Frac2
{
template <typename T>
auto strip_const(const T* t) noexcept
{
    return const_cast<T*>(t);
}

namespace kernels
{
static inline std::string commonDatatypes()
{
    static_assert(sizeof(UniformGridItem) == 4 * sizeof(uint32_t) + 1 * sizeof(int32_t),
                  "Incompatible size for OpenCL grid item");

    return "typedef struct {uint x; uint y;} u32x2;\n"
           "typedef struct {u32x2 size; uint stride;} image_size_t;\n"
           "typedef struct {u32x2 pos; u32x2 size; int category;} grid_item_t;\n"
           "typedef struct {image_size_t size; __global const uchar* data;} image_t;\n"
           "static uchar read_pixel(const image_t image, uint px, uint py) {"
           "    return image.data[px + py * image.size.stride];"
           "}\n";
}
}
}

#endif // CLCOMMON_HPP
