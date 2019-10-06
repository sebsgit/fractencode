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
	return is64Bit() ? "C:/Windows/SysWOW64/OpenCL.dll" : "C:/Windows/System32/OpenCL.dll";
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
		const auto grid = Frac2::createUniformGrid(testImage.size(), gridSize, gridOffset);
		BrightnessBlocksClassifier2 classifier(testImage, testImage);
		std::vector<int> categoriesCpu;
		for (auto & item : grid.items()) {
			categoriesCpu.push_back(classifier.getCategory(testImage, item));
		}
		REQUIRE(categoriesCpu.size() == grid.items().size());

		auto device = std::move(opencl_rt::gpu_devices()[0]);
		auto context = opencl_rt::context(device);
		auto queue = opencl_rt::command_queue(context, device);
		auto inputBuffer = context.create_buffer<uint8_t>(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * stride, testImage.data());
		REQUIRE(inputBuffer.handle());

		const size_t gridItemCount = grid.items().size();
		const size_t sumCount = gridItemCount * 4;
		auto outputSumBuffer = context.create_buffer<int>(CL_MEM_READ_WRITE, sumCount, nullptr);
		auto resultBuffer = context.create_buffer<int>(CL_MEM_WRITE_ONLY, sumCount / 4, nullptr);

		struct grid_item_t {
			cl_uint x, y, w, h;
		};

		struct image_size_t {
			cl_uint w = 0, h = 0, stride = 0;
		};

		const std::string createSums = 
			"typedef struct {uint x, y, w, h;} grid_item_t;"
			"typedef struct { uint w, h, stride; } image_size_t;"
			"\n"
			"static uchar read_pixel(__global const uchar* data, const image_size_t size, uint px, uint py) {"
			"    return data[px + py * size.stride];"
			"}"
			"\n"
			"static int sum_pixels(__global const uchar* data, const image_size_t size, const grid_item_t p, int block_id) {"
			"    int result = 0;"
			"    uint i, j;"
			"    uint x = p.x;"
			"    uint y = p.y;"
			"    uint w = p.w / 2;"
			"    uint h = p.h / 2;"
			"    if (block_id == 1) x += w;"
			"    else if (block_id == 2) y += h;"
			"    else if (block_id == 3) { x += w; y += h; }"
			"    for (i=0 ; i<h ; ++i) {"
			"        for (j=0 ; j<w ; ++j) {"
			"            result += read_pixel(data, size, x+j, y+i);"
			"        }"
			"    }"
			"    return result;"
			"}"
			"\n"
			"__kernel void sum_blocks(const image_size_t image_size, "
			"        __global const uchar* data,"
			"        __global const grid_item_t* grid,"
			"        uint grid_size,"
			"        __global int* output)"
			"{"
			"    const int sum_index = get_global_size(0) * get_global_id(1) + get_global_id(0);"
			"    const int grid_item_id = sum_index / 4;"
			"    const int block_id = sum_index % 4;"
			"    if (grid_item_id < grid_size) { "
			"        output[sum_index] = sum_pixels(data, image_size, grid[grid_item_id], block_id);"
			"    }"
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
			"__kernel void create_categories(__global const int* input, __global int* output) {"
			"    const int output_index = get_global_size(0) * get_global_id(1) + get_global_id(0);"
			"    const int x = output_index * 4;"
			"    output[output_index] = get_category(input[x], input[x + 1], input[x + 2], input[x + 3]);"
			"}";
		auto program = opencl_rt::program(context, createSums);
		try {
			program.build("-w -Werror");
		}
		catch (const std::exception& exc) {
			std::cout << program.build_log(device) << ' ' << exc.what();
			REQUIRE(false);
		}

		std::vector<grid_item_t> grid_gpu;
		for (auto & it : grid.items()) {
			grid_gpu.push_back(grid_item_t{it.origin.x(), it.origin.y(), it.size.x(), it.size.y()});
		}

		auto gridBuffer = context.create_buffer<grid_item_t>(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, grid_gpu.size(), grid_gpu.data());

		auto sumKernel = program.kernel("sum_blocks");
		sumKernel.set_args(image_size_t{ width, height, stride },
			inputBuffer.handle(),
			gridBuffer.handle(),
			grid_gpu.size(),
			outputSumBuffer.handle());

		auto categoryKernel = program.kernel("create_categories");
		categoryKernel.set_args(outputSumBuffer.handle(),
			resultBuffer.handle());

		opencl_rt::event sum_kernel_done;
		opencl_rt::event category_kernel_done;
		queue.enqueue(sumKernel, std::array<size_t, 2>{sumCount / 4, 4}, std::array<size_t, 2>{4, 4}, &sum_kernel_done);
		queue.enqueue(categoryKernel, std::array<size_t, 2>{gridItemCount / 32, 32}, std::array<size_t, 2>{4, 8}, &category_kernel_done, std::array<cl_event, 1>{sum_kernel_done.handle()});

		std::vector<int> categoriesGpu;
		categoriesGpu.resize(gridItemCount);
		opencl_rt::event read_done;
		queue.enqueue_non_blocking_read(resultBuffer, 0, resultBuffer.element_size() * gridItemCount, categoriesGpu.data(), read_done, std::array<cl_event, 1> {category_kernel_done.handle()});
		read_done.wait();

		REQUIRE(categoriesCpu.size() == categoriesGpu.size());
		for (size_t i = 0; i < categoriesGpu.size(); ++i) {
			if (categoriesCpu[i] != categoriesGpu[i])
				std::cout << i << ':' << categoriesCpu[i] << '/' << categoriesGpu[i] << '\n';
			REQUIRE(categoriesCpu[i] == categoriesGpu[i]);
		}
	}
}
#endif
