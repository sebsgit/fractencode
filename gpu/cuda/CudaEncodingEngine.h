#pragma once
#include "encode/EncodingEngine.h"

#include "CudaConf.h"

#ifdef FRAC_WITH_CUDA

namespace Frac {
	class CudaEncodingEngine : public AbstractEncodingEngine {
	public:
		CudaEncodingEngine(const encode_parameters_t& params, const Image& image, const PartitionPtr& sourceGrid)
			: AbstractEncodingEngine(params, image, sourceGrid)
		{ }
		~CudaEncodingEngine();
		void init() override;
		void finalize() override;
	protected:
		encode_item_t encode_impl(const PartitionItemPtr& target) const override;
	private:
		class CudaEncoderBackend;
		std::shared_ptr<CudaEncoderBackend> _backend;
	};
}

#endif
