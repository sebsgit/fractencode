#include "OpenCLEncodingEngine.hpp"
#include <CL/cl.h>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

using namespace Frac2;

struct platform_gpu_info {
    cl_platform_id platformId = nullptr;
    std::vector<cl_device_id> gpuId;
};

static std::vector<platform_gpu_info> findGpuDevices()
{
    std::vector<platform_gpu_info> result;
    std::vector<cl_platform_id> platforms;
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);
    platforms.resize(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    for (auto id : platforms) {
        platform_gpu_info info;
        info.platformId = id;
        std::string value;
        std::vector<cl_device_id> devices;
        cl_uint deviceCount = 0;
        clGetDeviceIDs(id, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
        if (deviceCount > 0) {
            devices.resize(deviceCount);
            clGetDeviceIDs(id, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr);
            info.gpuId = devices;
            result.push_back(info);
        }
    }
    return result;
}

static std::string getClDeviceInfoString(cl_device_id id, cl_device_info info)
{
    std::string result;
    size_t resultSize = 0;
    clGetDeviceInfo(id, info, 0, nullptr, &resultSize);
    result.resize(resultSize);
    clGetDeviceInfo(id, info, resultSize, &result[0], nullptr);
    return result;
}

static std::vector<size_t> getMaximumWorkGroupSizes(cl_device_id id)
{
    std::vector<size_t> result;
    size_t maxDimensions = 0;
    clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxDimensions), &maxDimensions, nullptr);
    result.resize(maxDimensions);
    clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(result[0]) * maxDimensions, result.data(), nullptr);
    return result;
}

namespace {
    struct kernel_work_item_result_t {
        uint32_t x;
        uint32_t y;
        uint32_t w;
        uint32_t h;
        double distance = 100000.0;
        double contrast = 0.0;
        double brightness = 0.0;
        int32_t transformType = 0;
        uint32_t sourceX = 0;
        uint32_t sourceY = 0;
        uint32_t sourceWidth = 0;
        uint32_t sourceHeight = 0;
    };
}

struct OpenCLEncodingEngine::Priv {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_mem imageBuffer = nullptr;
    cl_mem partitionBuffer = nullptr;
    cl_mem resultBuffer = nullptr;
    cl_kernel kernelID = nullptr;
    std::vector<size_t> workSizes;

    ~Priv() {
        if (this->kernelID)
            clReleaseKernel(this->kernelID);
        if (this->imageBuffer)
            clReleaseMemObject(this->imageBuffer);
        if (this->partitionBuffer)
            clReleaseMemObject(this->partitionBuffer);
        if (this->resultBuffer)
            clReleaseMemObject(this->resultBuffer);
        if (this->queue)
            clReleaseCommandQueue(this->queue);
        if (this->context)
            clReleaseContext(this->context);
    }
};

OpenCLEncodingEngine::OpenCLEncodingEngine(const encode_parameters_t& params,
    const ImagePlane& sourceImage,
    const UniformGrid& sourceGrid)
    :AbstractEncodingEngine2(params, sourceImage, sourceGrid)
    ,_d(new Priv())
{

}

OpenCLEncodingEngine::~OpenCLEncodingEngine() = default;

void OpenCLEncodingEngine::init()
{
    //TODO: for now get the first gpu device
    auto platforms = findGpuDevices();
    cl_platform_id platformID = {};
    cl_device_id deviceID = {};
    for (const auto& platform : platforms) {
        if (platformID == nullptr)
            for (const auto& gpu : platform.gpuId) {
                auto name = getClDeviceInfoString(gpu, CL_DEVICE_NAME);
                std::cout << "Using CL device: " << name << '\n';
                platformID = platform.platformId;
                deviceID = gpu;
                break;
            }
    }
    cl_context_properties contextParams[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<intptr_t>(platformID), 0};
    cl_int errorCode = 0;
    this->_d->context = clCreateContext(contextParams, 1, &deviceID, nullptr, nullptr, &errorCode);
    if (errorCode != CL_SUCCESS) {
        throw std::runtime_error("Cannot create CL context");
    }
    this->_d->queue = clCreateCommandQueue(this->_d->context, deviceID, 0, &errorCode);
    if (errorCode != CL_SUCCESS) {
        throw std::runtime_error("Cannot create command queue");
    }
    this->_d->imageBuffer = clCreateBuffer(this->_d->context, CL_MEM_COPY_HOST_PTR, this->_image.sizeInBytes(), (void*)this->_image.data(), &errorCode);
    if (errorCode != CL_SUCCESS) {
        throw std::runtime_error("Cannot upload image to the device");
    }
    const auto& partitionItems = this->_source.items();
    size_t partitionSizeInBytes = partitionItems.size() * sizeof(partitionItems[0]);
    this->_d->partitionBuffer = clCreateBuffer(this->_d->context, CL_MEM_COPY_HOST_PTR, partitionSizeInBytes, (void*)partitionItems.data(), &errorCode);
    if (errorCode != CL_SUCCESS) {
        throw std::runtime_error("Cannot upload partition data to the device");
    }
    this->_d->resultBuffer = clCreateBuffer(this->_d->context, CL_MEM_WRITE_ONLY, sizeof(kernel_work_item_result_t) * partitionItems.size(), nullptr, &errorCode);
    if (errorCode != CL_SUCCESS) {
        throw std::runtime_error("Cannot allocate result memory buffer on the device");
    }
    auto maxWorkSizes = getMaximumWorkGroupSizes(deviceID);
    std::cout << "maximum work group sizes: ";
    std::copy(maxWorkSizes.begin(), maxWorkSizes.end(), std::ostream_iterator<size_t>(std::cout, ", "));
    std::cout << '\n';
    size_t maxItems = std::accumulate(maxWorkSizes.begin(), maxWorkSizes.end(), 1, std::multiplies());
    std::cout << maxItems << ' ' << partitionItems.size() << '\n';
    if (maxItems < partitionItems.size()) {
        std::cout << "partition does not fit into single kernel launch\n";
        //TODO: schedule multiple kernels
    }
    //TODO: optimal size
    this->_d->workSizes = maxWorkSizes;

#define KERNEL_SOURCE(what) #what
    const std::string kernel = KERNEL_SOURCE(
        typedef struct {
            uint x;
            uint y;
        } u32pair_t;
        
        typedef struct {
            u32pair_t pt;
            u32pair_t size;
        } partition_item_t;

        typedef struct {
            uint x;
            uint y;
            uint w;
            uint h;
            double distance;
            double contrast;
            double brightness;
            int transformType;
            uint sourceX;
            uint sourceY;
            uint sourceWidth;
            uint sourceHeight;
        } kernel_work_item_result_t;

        static void map_coords(int type, uint* rx, uint* ry, const uint x, const uint y, const uint sx, const uint sy) {
            switch (type) {
            case 0:
                // { 1, 0, 0, 0,  0, 1, 0, 0 }
                *rx = x;
                *ry = y;
                break;
            case 1:
                // { 0, 1, 0, 0,  -1, 0, 1, 0 }
                *rx = 0 * x + 1 * y + 0 * (sx - 1) + 0 * (sy - 1);
                *ry = -1 * x + 0 * y + 1 * (sx - 1) + 0 * (sy - 1);
                break;
            case 2:
                // { -1, 0, 1, 0,  0, -1, 0, 1 }
                *rx = -1 * x + 0 * y + 1 * (sx - 1) + 0 * (sy - 1);
                *ry = 0 * x + -1 * y + 0 * (sx - 1) + 1 * (sy - 1);
                break;
            case 3:
                // { 0, -1, 0, 1,  1, 0, 0, 0 }
                *rx = 0 * x + -1 * y + 0 * (sx - 1) + 1 * (sy - 1);
                *ry = 1 * x + 0 * y + 0 * (sx - 1) + 0 * (sy - 1);
                break;
            case 4:
                // { 1, 0, 0, 0,   0, -1, 0, 1 }
                *rx = 1 * x + 0 * y + 0 * (sx - 1) + 0 * (sy - 1);
                *ry = 0 * x + -1 * y + 0 * (sx - 1) + 1 * (sy - 1);
                break;
            case 5:
                // { 0, 1, 0, 0,   1, 0, 0, 0 }
                *rx = 0 * x + 1 * y + 0 * (sx - 1) + 0 * (sy - 1);
                *ry = 1 * x + 0 * y + 0 * (sx - 1) + 0 * (sy - 1);
                break;
            case 6:
                // { -1, 0, 1, 0,  0, 1, 0, 0 }
                *rx = -1 * x + 0 * y + 1 * (sx - 1) + 0 * (sy - 1);
                *ry = 0 * x + 1 * y + 0 * (sx - 1) + 0 * (sy - 1);
                break;
            case 7:
                // { 0, -1, 0, 1, -1, 0, 1, 0 }
                *rx = 0 * x + -1 * y + 0 * (sx - 1) + 1 * (sy - 1);
                *ry = -1 * x + 0 * y + 1 * (sx - 1) + 0 * (sy - 1);
                break;
            default:
                break;
            }
        }

        
        static u32pair_t transform_patch(int type,
            uint local_x, uint local_y,
            uint patch_offset_x, uint patch_offset_y,
            uint patch_width, uint patch_height)
        {   
            uint x;
            uint y;
            map_coords(type, &x, &y, local_x, local_y, patch_width, patch_height);
            u32pair_t result;
            result.x = x + patch_offset_x;
            result.y = y + patch_offset_y;
            return result;
        }

        __kernel void encode_item(__global const uchar* image, int width, int height, int stride, 
                                    __global const partition_item_t* partition, int partition_size,
                                    __global kernel_work_item_result_t* result) 
        {
            int index = get_global_id(2) * get_global_size(1) * get_global_size(0) + get_global_id(1) * get_global_size(0) + get_global_id(0);
            if (index < partition_size) {
                result[index].x = index;
                result[index].distance = 0.45 * index + 4.12;
            }
        }
    );
#undef KERNEL_SOURCE

    auto sourceStr = kernel.c_str();
    auto program = clCreateProgramWithSource(this->_d->context, 1, (const char**)&sourceStr, nullptr, &errorCode);

    if (errorCode != CL_SUCCESS) {
        throw std::runtime_error("Cannot create program!");
    }
    errorCode = clBuildProgram(program, 1, &deviceID, "-Werror", nullptr, nullptr);
    if (errorCode != CL_SUCCESS) {
        size_t resultSize = 0;
        clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, nullptr, &resultSize);
        std::string log(resultSize + 1, '\0');
        clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, resultSize, (void*)log.c_str(), nullptr);
        std::cout << "Program log: " << log << '\n';
        throw std::runtime_error("Cannot build program!");
    }
    this->_d->kernelID = clCreateKernel(program, "encode_item", &errorCode);
    if (errorCode != CL_SUCCESS) {
        throw std::runtime_error("Cannot create kernel!");
    }
}

void OpenCLEncodingEngine::finalize()
{
    this->_d.reset();
}

encode_item_t OpenCLEncodingEngine::encode_impl(const UniformGridItem& targetItem) const
{
    /*
     encode_item(__global const uchar* image, int width, int height, int stride, 
                                    __global const partition_item_t* partition, int partition_size,
                                    __global kernel_work_item_result_t* result) 
    */
    int32_t width = this->_image.width();
    int32_t height = this->_image.height();
    int32_t stride = this->_image.stride();
    int32_t partitionSize = static_cast<int32_t>(this->_source.items().size());
    clSetKernelArg(this->_d->kernelID, 0, sizeof(this->_d->imageBuffer), &this->_d->imageBuffer);
    clSetKernelArg(this->_d->kernelID, 1, sizeof(width), &width);
    clSetKernelArg(this->_d->kernelID, 2, sizeof(height), &height);
    clSetKernelArg(this->_d->kernelID, 3, sizeof(stride), &stride);
    clSetKernelArg(this->_d->kernelID, 4, sizeof(this->_d->partitionBuffer), &this->_d->partitionBuffer);
    clSetKernelArg(this->_d->kernelID, 5, sizeof(partitionSize), &partitionSize);
    clSetKernelArg(this->_d->kernelID, 6, sizeof(this->_d->resultBuffer), &this->_d->resultBuffer);

    cl_event event;
    cl_int errorCode = clEnqueueNDRangeKernel(this->_d->queue, this->_d->kernelID, this->_d->workSizes.size(), nullptr, this->_d->workSizes.data(), nullptr, 0, nullptr, &event);
    if (errorCode != CL_SUCCESS) {
        throw std::runtime_error("cannot execute kernel!");
    }
    clWaitForEvents(1, &event);

    std::vector<kernel_work_item_result_t> result(partitionSize);
    errorCode = clEnqueueReadBuffer(this->_d->queue, this->_d->resultBuffer, CL_TRUE, 0, partitionSize * sizeof(kernel_work_item_result_t),
        result.data(), 0, nullptr, nullptr);
    if (errorCode != CL_SUCCESS) {
        throw std::runtime_error("cannot read the result from kernel!");
    }
    for (auto & res : result) {
        std::cout << "x: " << res.distance << '\n';
    }
    return encode_item_t();
}
