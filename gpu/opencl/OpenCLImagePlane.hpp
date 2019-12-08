#ifndef OPENCLIMAGEPLANE_HPP
#define OPENCLIMAGEPLANE_HPP

#include "image/Image2.hpp"
#include "opencl/opencl_rt.hpp"
#include "common.hpp"

namespace Frac2 {
class OpenCLImagePlane
{
public:
    struct image_size_t {
        cl_uint w = 0, h = 0, stride = 0;
    };

    explicit OpenCLImagePlane(opencl_rt::context& context, const Frac2::ImagePlane& image)
        :context_(context)
        ,data_(context.create_buffer<uint8_t>(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, image.width() * image.stride(), image.data()))
        ,size_{image.width(), image.height(), image.stride()}
    {
    }
    auto handle() const noexcept { return data_.handle(); }
    auto size() const noexcept { return size_; }
    auto & context() const noexcept { return context_; }
private:
    opencl_rt::context& context_;
    opencl_rt::buffer<uint8_t> data_;
    const image_size_t size_;
};
}

#endif // OPENCLIMAGEPLANE_HPP
