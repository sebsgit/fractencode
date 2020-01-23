#ifndef OPENCLGRIDPARTITION_HPP
#define OPENCLGRIDPARTITION_HPP

#include "common.hpp"
#include "utils/opencl/opencl_rt.hpp"

namespace Frac2
{
class OpenCLGridPartition
{
public:
    explicit OpenCLGridPartition(opencl_rt::context& context, const Frac2::UniformGrid& grid)
        :context_(context)
        ,buffer_(context.create_buffer<UniformGridItem>(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, grid.items().size(), strip_const(grid.items().data())))
        ,count_(static_cast<cl_uint>(grid.items().size()))
    {
    }
    auto handle() const noexcept { return buffer_.handle(); }
    cl_uint size() const noexcept { return count_; }
    auto & context() const noexcept { return context_; }

    template <size_t InputEventCount = 0>
    [[nodiscard]] opencl_rt::event copy(opencl_rt::command_queue& queue, Frac2::UniformGrid& output, const std::array<cl_event, InputEventCount>& inputEvents = {})
    {
        opencl_rt::event result;
        queue.enqueue_non_blocking_read(buffer_, 0, count_ * buffer_.element_size(), strip_const(output.items().data()), result, inputEvents);
        return result;
    }
private:
    opencl_rt::context & context_;
    opencl_rt::buffer<UniformGridItem> buffer_;
    const cl_uint count_;
};
}

#endif // OPENCLGRIDPARTITION_HPP
