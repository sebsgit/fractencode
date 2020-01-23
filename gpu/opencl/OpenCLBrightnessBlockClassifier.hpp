#ifndef OPENCLBRIGHTNESSBLOCKCLASSIFIER_HPP
#define OPENCLBRIGHTNESSBLOCKCLASSIFIER_HPP

#include "utils/opencl/opencl_rt.hpp"
#include "common.hpp"
#include "OpenCLGridPartition.hpp"
#include "OpenCLImagePlane.hpp"

namespace Frac2
{
class OpenCLBrightnessBlockClassifier
{
public:
    explicit OpenCLBrightnessBlockClassifier(OpenCLGridPartition& partition)
        :partition_(partition)
    {
        const std::string createSums =
            kernels::commonDatatypes() +
            "\n"
            "static int sum_pixels(const image_t image, const grid_item_t p, int block_id) {"
            "    int result = 0;"
            "    uint i, j;"
            "    uint x = p.pos.x;"
            "    uint y = p.pos.y;"
            "    uint w = p.size.x / 2;"
            "    uint h = p.size.y / 2;"
            "    if (block_id == 1) x += w;"
            "    else if (block_id == 2) y += h;"
            "    else if (block_id == 3) { x += w; y += h; }"
            "    for (i=0 ; i<h ; ++i) {"
            "        for (j=0 ; j<w ; ++j) {"
            "            result += read_pixel(image, x+j, y+i);"
            "        }"
            "    }"
            "    return result;"
            "}"
            "\n"
            "int sum_block(const image_t image,"
            "        __global const grid_item_t* grid,"
            "        uint grid_size,"
            "        uint grid_item_id,"
            "        uint block_id)"
            "{"
            "    if (grid_item_id < grid_size) { "
            "        return sum_pixels(image, grid[grid_item_id], block_id);"
            "    }"
            "    return 0;"
            "}"
            "\n"
            "static int get_category(int a1, int a2, int a3, int a4) {"
            "const bool a1a2 = a1 > a2;"
            "const bool a1a3 = a1 > a3;"
            "const bool a1a4 = a1 > a4;"
            "const bool a2a1 = a2 > a1;"
            "const bool a2a3 = a2 > a3;"
            "const bool a2a4 = a2 > a4;"
            "const bool a3a1 = a3 > a1;"
            "const bool a3a2 = a3 > a2;"
            "const bool a3a4 = a3 > a4;"
            "const bool a4a1 = a4 > a1;"
            "const bool a4a2 = a4 > a2;"
            "const bool a4a3 = a4 > a3;"
                ""
            "if (a1a2 && a2a3 && a3a4) return 0;"
            "if (a3a1 && a1a4 && a4a2) return 0;"
            "if (a4a3 && a3a2 && a2a1) return 0;"
            "if (a2a4 && a4a1 && a1a3) return 0;"
                ""
            "if (a1a3 && a3a2 && a2a4) return 1;"
            "if (a2a1 && a1a4 && a4a3) return 1;"
            "if (a4a2 && a2a3 && a3a1) return 1;"
            "if (a3a4 && a4a1 && a1a2) return 1;"
                ""
            "if (a1a4 && a4a3 && a3a2) return 2;"
            "if (a4a1 && a1a2 && a2a3) return 2;"
            "if (a3a2 && a2a4 && a4a1) return 2;"
            "if (a2a3 && a3a1 && a1a4) return 2;"
                ""
            "if (a1a2 && a2a4 && a4a3) return 3;"
            "if (a3a1 && a1a2 && a2a4) return 3;"
            "if (a4a3 && a3a1 && a1a2) return 3;"
            "if (a2a4 && a4a3 && a3a1) return 3;"
                ""
            "if (a2a1 && a1a3 && a3a4) return 4;"
            "if (a1a3 && a3a4 && a4a2) return 4;"
            "if (a3a4 && a4a2 && a2a1) return 4;"
            "if (a4a2 && a2a1 && a1a3) return 4;"
                ""
            "if (a1a4 && a4a2 && a2a3) return 5;"
            "if (a4a1 && a1a3 && a3a4) return 5;"
            "if (a2a3 && a3a4 && a4a1) return 5;"
            "if (a3a2 && a2a1 && a1a4) return 5;"
            ""
            "return -1;"
            "}"
            "\n"
            "__kernel void classify(__global grid_item_t * grid, const uint grid_size, __global const uchar* data, const image_size_t size) {"
            "    const int idx = get_global_size(0) * get_global_id(1) + get_global_id(0);"
            "    image_t image; image.data = data; image.size = size;"
            "    int block_0 = sum_block(image, grid, grid_size, idx, 0);"
            "    int block_1 = sum_block(image, grid, grid_size, idx, 1);"
            "    int block_2 = sum_block(image, grid, grid_size, idx, 2);"
            "    int block_3 = sum_block(image, grid, grid_size, idx, 3);"
            "    grid[idx].category = get_category(block_0, block_1, block_2, block_3);"
            "}";
        classifyProgram_ = opencl_rt::program(partition.context(), createSums);
        classifyProgram_.build("-w -Werror");
        classifyKernel_ = classifyProgram_.create_kernel("classify");
    }
    [[nodiscard]] opencl_rt::event classify(opencl_rt::command_queue& queue, const OpenCLImagePlane& image)
    {
        opencl_rt::event result;
        const cl_uint gridItemCount = partition_.size();
        classifyKernel_.set_arg(0, partition_.handle());
        classifyKernel_.set_arg(1, gridItemCount);
        classifyKernel_.set_arg(2, image.handle());
        classifyKernel_.set_arg(3, image.size());
        queue.enqueue(classifyKernel_, std::array<size_t, 2>{gridItemCount / 4, 4}, std::array<size_t, 2>{4, 4}, &result);
        return result;
    }
private:
    OpenCLGridPartition & partition_;
    opencl_rt::program classifyProgram_;
    opencl_rt::kernel classifyKernel_;
};
}

#endif // OPENCLBRIGHTNESSBLOCKCLASSIFIER_HPP
