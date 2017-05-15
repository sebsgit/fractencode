#ifndef ENCODER_H
#define ENCODER_H

#include "image/partition.h"
#include "image/transform.h"
#include "image/sampler.h"
#include "image/partition/gridpartition.h"
#include "image/partition/presampledpartition.h"
#include "encode/TransformEstimator.h"
#include "encode/EncodingEngine.h"
#include "transformmatcher.h"
#include "classifier.h"
#include "edgeclassifier.h"
#include <iostream>
#include <sstream>

namespace Frac {
class DummyReporter : public ProgressReporter
{
public:
	void log(size_t, size_t) override {}
};

class Encoder {
public:
	struct encode_stats_t {
		uint64_t rejectedMappings = 0;
		uint64_t totalMappings = 0;

		void print() {
			std::cout << "classifier rejected " << rejectedMappings << " out of " << totalMappings << " comparisons (" << (100.0*rejectedMappings) / totalMappings << ")%\n";
		}
	};

public:
	Encoder(const Image& image, const encode_parameters_t& p, const PartitionCreator& sourceCreator, const PartitionCreator& targetCreator, ProgressReporter* reporter = nullptr)
		: _encodeParameters(p)
		, _metric(new RootMeanSquare())
		, _reporter(reporter ? reporter : new DummyReporter())
	{
		auto gridSource = sourceCreator.create(image);
		auto gridTarget = targetCreator.create(image);
		auto classifier = std::shared_ptr<ImageClassifier>(new CombinedClassifier(new BrightnessBlockClassifier, new ThresholdClassifier));
		if (p.noclassifier)
			classifier.reset(new DummyClassifier);
		this->_estimator.reset(new TransformEstimator(classifier, std::make_shared<TransformMatcher>(*_metric, p.rmsThreshold, p.sMax), gridSource));
		this->_engine.reset(new EncodingEngineCore(_encodeParameters, image, gridSource, _estimator, _reporter.get()));
		this->_engine->encode(gridTarget);
		this->_stats.totalMappings = gridSource->size() * gridTarget->size();
	}
	grid_encode_data_t data() const {
		this->_stats.rejectedMappings = this->_estimator->rejectedMappings();
		this->_stats.print();
		return this->_engine->result();
	}
private:
	const encode_parameters_t _encodeParameters;
	mutable encode_stats_t _stats;
	std::shared_ptr<TransformEstimator> _estimator;
	std::shared_ptr<Metric> _metric;
	std::unique_ptr<EncodingEngineCore> _engine;
	std::unique_ptr<ProgressReporter> _reporter;
};

class Decoder {
public:
	struct decode_stats_t {
		int iterations;
		double rms;
	};
	Decoder(Image& target, const int nMaxIterations = -1, const double rmsEpsilon = 0.00001, bool saveDecodeSteps = false)
		:_target(target)
		,_iterations(nMaxIterations < 0 ? 300 : nMaxIterations)
		,_rmsEpsilon(rmsEpsilon)
		,_saveSteps(saveDecodeSteps)
	{
	}
	decode_stats_t decode(const grid_encode_data_t& data) {
		AbstractBufferPtr<Image::Pixel> buffer = Buffer<Image::Pixel>::alloc(_target.height() * _target.width());
		buffer->memset(100);
		Image source(buffer, _target.width(), _target.height(), _target.stride());
		RootMeanSquare metric;
		int i=0;
		double rms = 0.0;
		if (_saveSteps)
			source.savePng("decode_debug0.png");
		for ( ; i<_iterations ; ++i) {
			this->decodeStep(source, _target, data);
			if (_saveSteps) {
				std::stringstream ss;
				ss << i + 1;
				_target.savePng("decode_debug" + ss.str() +".png");
			}
			rms = metric.distance(source, _target);
			if (rms < _rmsEpsilon)
				break;
			source = _target.copy();
		}
		return { i, rms };
	}
private:
	void decodeStep(const Image& source, Image& target, const grid_encode_data_t& data) const {
		for (uint32_t p = 0 ; p<data.encoded.size() ; ++p) {
			const encode_item_t enc = data.encoded.at(p);
			const item_match_t match = enc.match;
			Image sourcePart = source.slice(match.x, match.y, match.sourceItemSize.x(), match.sourceItemSize.y());
			Image targetPart = target.slice(enc.x, enc.y, enc.w, enc.h);
			Transform t = Transform(match.score.transform);
			t.copy(sourcePart, targetPart, match.score.contrast, match.score.brightness);
		}
	}
private:
	Image& _target;
	const int _iterations;
	const double _rmsEpsilon;
	const bool _saveSteps = false;
};

}

#endif // ENCODER_H
