#ifdef FRAC_TESTS
#include "catch.hpp"
#include "utils/opencl/opencl_rt.hpp"
#include "encode/Classifier2.hpp"
#include "image/Image2.hpp"
#include "image/partition2.hpp"
#include "gpu/opencl/OpenCLImagePlane.hpp"
#include "gpu/opencl/OpenCLGridPartition.hpp"
#include "gpu/opencl/OpenCLBrightnessBlockClassifier.hpp"
#include "gpu/opencl/common.hpp"

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

TEST_CASE("OpenCL", "[gpu][opencl]")
{
    SECTION("gpu detection")
    {
        REQUIRE(opencl_rt::load(openclLibPath()));
        REQUIRE_FALSE(opencl_rt::gpu_devices().empty());
    }
	SECTION("sizes of kernel datatypes")
	{
		auto device = opencl_rt::gpu_devices().front();
		auto ctx = opencl_rt::context(device);
		const auto programSource = Frac2::kernels::commonDatatypes()
			+ "\n"
			+ "__kernel void get_sizes(__global uint* output) {\n"
			+ "    output[0] = sizeof(u32x2);\n"
			+ "    output[1] = sizeof(image_size_t);\n"
			+ "    output[2] = sizeof(grid_item_t);\n"
			+ "    output[3] = sizeof(image_t);\n"
			+ "}";
		auto program = opencl_rt::program(ctx, programSource);
		REQUIRE_NOTHROW(program.build());
		auto kernel = program.create_kernel("get_sizes");
		auto buffer = ctx.create_buffer<uint32_t>(CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY, 4, nullptr);
		kernel.set_arg(0, buffer.handle());

		auto queue = opencl_rt::command_queue(ctx, device);
		REQUIRE_NOTHROW(queue.enqueue(kernel, std::array<size_t, 2>{1, 1}, std::array<size_t, 2>{1, 1}));
		queue.finish();

		uint32_t output[4];
		queue.enqueue_blocking_read(buffer, 0, 4 * sizeof(*output), output);

		REQUIRE(output[0] == sizeof(Size32u));
		REQUIRE(output[1] == sizeof(OpenCLImagePlane::image_size_t));
		REQUIRE(output[2] == sizeof(UniformGridItem));
		REQUIRE(output[3] == sizeof(OpenCLImagePlane::image_size_t) + sizeof(const uint8_t*));
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

		for (size_t i = 0; i < gpuGrid.items().size(); ++i)
		{
			REQUIRE(gpuGrid.items()[i].size == grid.items()[i].size);
			REQUIRE(gpuGrid.items()[i].origin == grid.items()[i].origin);
			REQUIRE(gpuGrid.items()[i].data.bb_classifierBin == grid.items()[i].data.bb_classifierBin);
		}
	}
}
#endif
