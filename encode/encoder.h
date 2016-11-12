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

struct classifier_data_t {
	PartitionItemPtr targetItem;	// item from target partition
	std::vector<PartitionItemPtr> candidates;	// items from source partition that could be a possible match for target item
};

class ClassifierNode {
public:
	using classifier_callback_t = std::function<bool(const classifier_data_t&)>;
	ClassifierNode(std::shared_ptr<ImageClassifier> classifier, classifier_callback_t callback)
		: _classifier(classifier)
		, _classifierCallback(callback) {}
	void classify(const Image& image, const PartitionPtr& sourcePartition, const PartitionPtr& targetPartition) {
		for (auto it : *targetPartition) {
			classifier_data_t result;
			result.targetItem = it;
			for (auto src : *sourcePartition) {
				if (this->_classifier->compare(it, src)) {
					result.candidates.push_back(src);
				}
			}
			if (!this->_classifierCallback(result))
				break;
		}
	}
private:
	std::shared_ptr<ImageClassifier> _classifier;
	 classifier_callback_t _classifierCallback;
};

class TransformMatcherNode {
public:
	TransformMatcherNode(std::shared_ptr<TransformMatcher> matcher, const PartitionPtr& sourcePartition, const PartitionPtr& targetPartition) 
		: _matcher(matcher)
		, _source(sourcePartition)
		, _target(targetPartition)
	{}
	void addCandidates(const classifier_data_t& data) {
		if (!data.candidates.empty()) {
			item_match_t result;
			for (auto src : data.candidates) {
				auto score = this->_matcher->match(data.targetItem, src);
				if (score.distance < result.score.distance) {
					result.score = score;
					result.x = src->pos().x();
					result.y = src->pos().y();
					result.sourceItemSize = src->sourceSize();
				}
				if (this->_matcher->checkDistance(result.score.distance))
					break;
			}
			encode_item_t enc;
			enc.x = data.targetItem->pos().x();
			enc.y = data.targetItem->pos().y();
			enc.w = data.targetItem->image().width();
			enc.h = data.targetItem->image().height();
			enc.match = result;
			this->_result.encoded.emplace_back(enc);
		}
	}
	grid_encode_data_t result() const {
		return this->_result;
	}
private:
	std::shared_ptr<TransformMatcher> _matcher;
	PartitionPtr _source;
	PartitionPtr _target;
	grid_encode_data_t _result; //TODO make it output partial results too
};

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
		
		TransformMatcherNode matcherNode(std::make_shared<TransformMatcher>(*_metric, p.rmsThreshold, p.sMax), gridSource, gridTarget);
		ClassifierNode classifierNode(this->_classifier, [&](const classifier_data_t& data) -> bool {
			matcherNode.addCandidates(data);
			return true;
		});
		classifierNode.classify(image, gridSource, gridTarget);
		this->_data = matcherNode.result();
		
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
