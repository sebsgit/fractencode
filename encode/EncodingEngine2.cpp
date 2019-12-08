#include "EncodingEngine2.hpp"
#include "utils/Assert.hpp"
#include "gpu/opencl/OpenCLEncodingEngine.hpp"

using namespace Frac2;

EncodingEngineCore2::EncodingEngineCore2(const encode_parameters_t& params, const ImagePlane& image, const UniformGrid& gridSource, const TransformEstimator2& estimator, ProgressReporter2* reporter)
    : _estimator(estimator)
    , _reporter(reporter)
{
    FRAC_ASSERT(reporter);
    const auto maxThreads = std::thread::hardware_concurrency();
    if (!params.nocpu)
        for (size_t i = 0; i < maxThreads; ++i) {
            auto engine = std::make_unique<CpuEncodingEngine2>(params, image, gridSource, this->_estimator);
            std::stringstream ss;
            ss << "cpu " << i;
            engine->setName(ss.str());
            this->_engines.push_back(std::move(engine));
        }
    //TODO: CUDA, OpenCL engines
    try {
     //   auto engine = std::make_unique<OpenCLEncodingEngine>(params, image, gridSource);
     //   engine->setName("OpenCL");
     //   this->_engines.push_back(std::move(engine));
    }
    catch (const std::exception & exc) {
        std::cout << "failed to create engine: " << exc.what();
    }
}
