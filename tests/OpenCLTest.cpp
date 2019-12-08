#ifdef FRAC_TESTS
#include "catch.hpp"
#include "utils/opencl/opencl_rt.hpp"
#include "encode/Classifier2.hpp"
#include "image/Image2.hpp"
#include "image/partition2.hpp"

using namespace Frac2;

static constexpr bool is64Bit() noexcept
{
	return sizeof(void*) > 4;
}

static std::string openclLibPath()
{
#ifdef __linux__
    return "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0";
#else
	return is64Bit() ? "C:/Windows/SysWOW64/OpenCL.dll" : "C:/Windows/System32/OpenCL.dll";
#endif
}

namespace {

template <typename T>
auto strip_const(const T* t) noexcept
{
    return const_cast<T*>(t);
}

namespace kernels
{
static std::string commonDatatypes()
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

TEST_CASE("OpenCL", "[gpu][opencl]")
{
    SECTION("gpu detection")
    {
        REQUIRE(opencl_rt::load(openclLibPath()));
        REQUIRE_FALSE(opencl_rt::gpu_devices().empty());
    }
    SECTION("bb classify opencl")
    {
        const uint32_t width = 512;
        const uint32_t height = 512;
        const uint32_t stride = width;
        ImagePlane testImage(Size32u(width, height), stride);
        for (uint32_t h = 0; h < height; ++h) {
            for (uint32_t w = 0; w < width; ++w) {
                testImage.setValue(w, h, (w * 11 + h * 43 + 124) % 256);
            }
        }
        const uint32_t blockSize = 4;
		const Size32u gridSize(blockSize, blockSize);
		const Size32u gridOffset(blockSize, blockSize / 2);
        BrightnessBlocksClassifier2 classifier(testImage, testImage);
        auto classifierCallback = [&](const Point2du& origin, const Size32u& size) {
            UniformGridItem::ExtraData data;
            classifier.preclassify(origin, size, data);
            return data;
        };
        const auto grid = Frac2::createUniformGrid(testImage.size(), gridSize, gridOffset, classifierCallback);
        REQUIRE_FALSE(std::all_of(grid.items().begin(), grid.items().end(), [](const UniformGridItem& item) {
            return item.data.bb_classifierBin == 0;
        }));

		auto device = std::move(opencl_rt::gpu_devices()[0]);
		auto context = opencl_rt::context(device);
		auto queue = opencl_rt::command_queue(context, device);
        auto inputBuffer = OpenCLImagePlane(context, testImage);
		REQUIRE(inputBuffer.handle());

        auto gridBuffer = OpenCLGridPartition(context, grid);
        auto gpuClassifier = OpenCLBrightnessBlockClassifier(gridBuffer);
        auto classifyEvent = gpuClassifier.classify(queue, inputBuffer);

        Frac2::UniformGrid gpuGrid = UniformGrid::createEmpty(grid.items().size());
        REQUIRE(gpuGrid.items().size() == grid.items().size());
        auto copyEvent = gridBuffer.copy(queue, gpuGrid, std::array<cl_event, 1>{classifyEvent.handle()});
        copyEvent.wait();
        REQUIRE(gpuGrid.items().size() == grid.items().size());

        for (size_t i=0 ; i<gpuGrid.items().size() ; ++i)
        {
            REQUIRE(gpuGrid.items()[i].size == grid.items()[i].size);
            REQUIRE(gpuGrid.items()[i].origin == grid.items()[i].origin);
            REQUIRE(gpuGrid.items()[i].data.bb_classifierBin == grid.items()[i].data.bb_classifierBin);
        }
    }
}
#endif
