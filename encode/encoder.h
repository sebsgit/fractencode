#ifndef ENCODER_H
#define ENCODER_H

#include "image/partition.h"
#include "image/transform.h"
#include "image/sampler.h"
#include "image/partition/gridpartition.h"
#include "image/partition/presampledpartition.h"
#include "transformmatcher.h"
#include "classifier.h"
#include "edgeclassifier.h"
#include <iostream>
#include <sstream>

namespace Frac {

// class encoder data

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
		: _metric(new RootMeanSquare)
		, _classifier(new CombinedClassifier(new BrightnessBlockClassifier, new ThresholdClassifier))
		, _encodeParameters(p)
		, _matcher(*_metric, p.rmsThreshold, p.sMax)
	{
		PartitionPtr gridSource = sourceCreator.create(image);
		PartitionPtr gridTarget = targetCreator.create(image);
		_data = gridTarget->estimateMapping(gridSource, *this->_classifier, this->_matcher, _stats.rejectedMappings);
		this->_stats.totalMappings = gridSource->size() * gridTarget->size();
		this->_stats.print();
	}
	grid_encode_data_t data() const {
		return _data;
	}
private:
	std::shared_ptr<Metric> _metric;
	std::shared_ptr<ImageClassifier> _classifier;
	grid_encode_data_t _data;
	const encode_parameters_t _encodeParameters;
	const TransformMatcher _matcher;
	mutable encode_stats_t _stats;
};

class Decoder {
public:
	struct decode_stats_t {
		int iterations;
		double rms;
	};
	Decoder(Image& target, const int nMaxIterations = -1, const double rmsEpsilon = 0.00001)
		:_target(target)
		,_iterations(nMaxIterations < 0 ? 300 : nMaxIterations)
		,_rmsEpsilon(rmsEpsilon)
	{
	}
	decode_stats_t decode(const grid_encode_data_t& data) {
		AbstractBufferPtr<Image::Pixel> buffer = Buffer<Image::Pixel>::alloc(_target.height() * _target.width());
		buffer->memset(100);
		Image source(buffer, _target.width(), _target.height(), _target.stride());
		RootMeanSquare metric;
		int i=0;
		double rms = 0.0;
		for ( ; i<_iterations ; ++i) {
			this->decodeStep(source, _target, data);
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
};

}

#endif // ENCODER_H
