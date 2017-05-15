#include "EncodingEngine.h"
#include "gpu/cuda/CudaEncodingEngine.h"

using namespace Frac;

EncodingEngineCore::EncodingEngineCore(const encode_parameters_t& params, const Image& image, const PartitionPtr& gridSource, const std::shared_ptr<TransformEstimator>& estimator, ProgressReporter* reporter)
	: _estimator(estimator)
	, _reporter(reporter)
{
	assert(estimator);
	assert(reporter);
	const auto maxThreads = std::thread::hardware_concurrency() - 1;
	if (!params.nocpu)
		for (size_t i = 0; i < maxThreads; ++i) {
			auto engine = std::unique_ptr<CpuEncodingEngine>(new CpuEncodingEngine(params, image, gridSource, estimator));
			std::stringstream ss;
			ss << "cpu " << i;
			engine->setName(ss.str());
			this->_engines.push_back(std::move(engine));
		}
#ifdef FRAC_WITH_CUDA
	if (!params.nogpu) {
		auto cudaEngine = std::unique_ptr<CudaEncodingEngine>(new CudaEncodingEngine(params, image, gridSource));
		cudaEngine->setName("gpu (cuda)");
		this->_engines.push_back(std::move(cudaEngine));
	}
#endif
}
