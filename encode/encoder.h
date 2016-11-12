#ifndef ENCODER_H
#define ENCODER_H

#include "image/partition.h"
#include "image/transform.h"
#include "image/sampler.h"
#include "image/partition/gridpartition.h"
#include "image/partition/presampledpartition.h"
#include "encode/TransformEstimator.h"
#include "transformmatcher.h"
#include "classifier.h"
#include "edgeclassifier.h"
#include <iostream>
#include <sstream>

namespace Frac {

class Encoder {
public:
	struct encode_parameters_t {
		int sourceGridSize = 16;
		int targetGridSize = 4;
		int latticeSize = 2;
		double rmsThreshold = 0.0;
		double sMax = -1.0;
	};

	struct encode_stats_t {
		uint64_t rejectedMappings = 0;
		uint64_t totalMappings = 0;

		void print() {
			std::cout << "classifier rejected " << rejectedMappings << " out of " << totalMappings << " comparisons (" << (100.0*rejectedMappings) / totalMappings << ")%\n";
		}
	};

public:
	Encoder(const Image& image, const encode_parameters_t& p, const PartitionCreator& sourceCreator, const PartitionCreator& targetCreator)
		: _encodeParameters(p)
	{
		auto gridSource = sourceCreator.create(image);
		auto gridTarget = targetCreator.create(image);
		auto metric = RootMeanSquare();
		auto classifier = std::shared_ptr<ImageClassifier>(new CombinedClassifier(new BrightnessBlockClassifier, new ThresholdClassifier));
		TransformEstimator transformEstimator(classifier, std::make_shared<TransformMatcher>(metric, p.rmsThreshold, p.sMax), gridSource, gridTarget);
		transformEstimator.estimate(image);
		this->_data = transformEstimator.result();
		this->_stats.rejectedMappings = transformEstimator.rejectedMappings();
		this->_stats.totalMappings = gridSource->size() * gridTarget->size();
		this->_stats.print();
	}
	grid_encode_data_t data() const {
		return _data;
	}
private:
	grid_encode_data_t _data;
	const encode_parameters_t _encodeParameters;
	mutable encode_stats_t _stats;
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
