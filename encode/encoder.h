#ifndef ENCODER_H
#define ENCODER_H

#include "image/partition.h"
#include "image/transform.h"
#include "image/sampler.h"
#include "image/partition/gridpartition.h"
#include "transformmatcher.h"
#include "classifier.h"
#include <iostream>
#include <sstream>

namespace Frac {

// class encoder data

class Encoder {
public:
    struct item_match_t {
        TransformMatcher::score_t score;
        uint32_t x = 0;
        uint32_t y = 0;
    };
    struct encode_item_t {
        uint32_t x, y, w, h;
        item_match_t match;
    };

    struct grid_encode_data_t {
        std::vector<encode_item_t> encoded;
        Size32u sourceItemSize;
    };

    struct encode_parameters_t {
        int sourceGridSize = 16;
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
    Encoder(const Image& image, const encode_parameters_t& p)
        : _metric(new RootMeanSquare)
        , _classifier(new TextureClassifier)
        , _encodeParameters(p)
        , _matcher(*_metric, p.rmsThreshold, p.sMax)
    {
        const Size32u gridSizeSource(p.sourceGridSize, p.sourceGridSize);
        const Size32u gridOffset = gridSizeSource / 2;
        const Size32u gridSizeTarget = gridSizeSource / 2;
        GridPartitionCreator gridCreatorSource(gridSizeSource, gridOffset);
        GridPartitionCreator gridCreatorTarget(gridSizeTarget, gridOffset);
        PartitionData gridSource = gridCreatorSource.create(image);
        PartitionData gridTarget = gridCreatorTarget.create(image);
        int debug = 0;
        for (auto it : gridTarget) {
            item_match_t match = this->matchItem(it, gridSource);
            std::cout << it->pos().x() << ", " << it->pos().y() << " --> " << match.x << ',' << match.y << " d: " << match.score.distance << "\n";
            std::cout << "s, o: " << match.score.contrast << ' ' << match.score.brightness << "\n";
            encode_item_t enc;
            enc.x = it->pos().x();
            enc.y = it->pos().y();
            enc.w = it->image().width();
            enc.h = it->image().height();
            enc.match = match;
            _data.encoded.push_back(enc);
            ++debug;
        }
        _data.sourceItemSize = gridSizeSource;

        this->_stats.totalMappings = gridSource.size() * gridTarget.size();
        this->_stats.print();
    }
    grid_encode_data_t data() const {
        return _data;
    }
private:
    item_match_t matchItem(const PartitionItemPtr& a, const PartitionData& data) const {
        item_match_t result;
        uint32_t i = 0;
        for (auto it : data) {
            if (this->_classifier->compare(a->image(), it->image())) {
                auto score = this->_matcher.match(a, it);
                if (score.distance < result.score.distance) {
                    result.score = score;
                    result.x = it->pos().x();
                    result.y = it->pos().y();
                }
                if (this->_matcher.checkDistance(result.score.distance))
                    break;
            } else {
                this->_stats.rejectedMappings++;
            }
            ++i;
        }
        return result;
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
    decode_stats_t decode(const Encoder::grid_encode_data_t& data) {
        AbstractBufferPtr<uint8_t> buffer = Buffer<uint8_t>::alloc(_target.height() * _target.width());
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
    void decodeStep(const Image& source, Image& target, const Encoder::grid_encode_data_t& data) const {
        for (uint32_t p = 0 ; p<data.encoded.size() ; ++p) {
            const Encoder::encode_item_t enc = data.encoded.at(p);
            const Encoder::item_match_t match = enc.match;
            Image sourcePart = source.slice(match.x, match.y, data.sourceItemSize.x(), data.sourceItemSize.y());
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
