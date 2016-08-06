#ifndef ENCODER_H
#define ENCODER_H

#include "image/partition.h"
#include "image/transform.h"
#include "image/sampler.h"
#include "classifier.h"
#include <iostream>
#include <sstream>

namespace Frac {

// class encoder data

class Encoder {
public:
    struct score_t {
        double distance = 100000.0;
        double contrast = 0.0;
        double brightness = 0.0;
        Transform::Type transform = Transform::Id;
    };
    struct item_match_t {
        score_t score;
        uint32_t x = 0;
        uint32_t y = 0;

        uint32_t i = 0;
    };
    struct grid_encode_data_t {
        std::vector<item_match_t> items;
        Size32u targetItemSize;
        Size32u sourceItemSize;
        Size32u offsetSize;
    };

public:
    Encoder(const Image& image, int gridSize = 16)
        : _metric(new RootMeanSquare)
        , _classifier(new DummyClassifier)
    {
        const Size32u gridSizeSource(gridSize, gridSize);
        const Size32u gridOffset = gridSizeSource / 2;
        const Size32u gridSizeTarget = gridSizeSource / 2;
        GridPartition gridCreatorSource(gridSizeSource, gridOffset);
        GridPartition gridCreatorTarget(gridSizeTarget, gridOffset);
        PartitionData gridSource = gridCreatorSource.create(image);
        PartitionData gridTarget = gridCreatorTarget.create(image);
        int debug = 0;
        for (auto it : gridTarget) {
            item_match_t match = this->matchItem(it, gridSource);
            std::cout << it->pos().x() << ", " << it->pos().y() << " --> " << match.x << ',' << match.y << " d: " << match.score.distance << "\n";
            std::cout << "s, o: " << match.score.contrast << ' ' << match.score.brightness << "\n";
            _data.items.push_back(match);
            ++debug;
        }
        _data.sourceItemSize = gridSizeSource;
        _data.targetItemSize = gridSizeTarget;
        _data.offsetSize = gridOffset;
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
                score_t score = this->matchTransform(a, it);
                if (score.distance < result.score.distance) {
                    result.score = score;
                    result.x = it->pos().x();
                    result.y = it->pos().y();
                    result.i = i;
                }
            }
            ++i;
        }
        return result;
    }
    score_t matchTransform(const PartitionItemPtr& a, const PartitionItemPtr& b) const {
        score_t result;
        Transform t(Transform::Id);
        const SamplerBilinear samplerB(b->image());
        do {
            score_t candidate;
            const double N = (double)(a->image().width()) * a->image().height();
            double sumA = 0.0, sumA2 = 0.0, sumB = 0.0, sumB2 = 0.0, sumAB = 0.0;
            for (uint32_t y = 0 ; y<a->image().height() ; ++y) {
                for (uint32_t x = 0 ; x<a->image().width() ; ++x) {
                    const auto srcY = (y * b->image().height()) / a->image().height();
                    const auto srcX = (x * b->image().width()) / a->image().width();
                    const double valA = convert<double, Image::Pixel>(a->image().data()->get()[x + y * a->image().stride()]);
                    const double valB = convert<double, Image::Pixel>(samplerB(srcX, srcY, t, b->image().size()));
                    sumA += valA;
                    sumB += valB;
                    sumA2 += valA * valA;
                    sumB2 += valB * valB;
                    sumAB += valA * valB;
                }
            }
            const double tmp = (N * sumA2 - sumA * sumA);
            const double s = (fabs(tmp) < 0.00001) ? 0.0 : (N * sumAB - sumA * sumB) / tmp;
            const double o = (sumB - s * sumA) / N;
            candidate.contrast = s;
            candidate.brightness = o;
            //auto D = (sumB2 + s * (s * sumA2 - 2 * sumAB + 2 * o * sumA) + o * ( N * o - 2 * sumB)) / N;//
            candidate.distance = _metric->distance(a->image(), b->image(), t);
            candidate.transform = t.type();
            if (candidate.distance <= result.distance) {
                result = candidate;
            }
        } while (t.next() != Transform::Id);
        return result;
    }
private:
    std::shared_ptr<Metric> _metric;
    std::shared_ptr<ImageClassifier> _classifier;
    grid_encode_data_t _data;
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
        GridPartition gridCreatorSource(data.sourceItemSize, data.offsetSize);
        GridPartition gridCreatorTarget(data.targetItemSize, data.offsetSize);
        PartitionData gridSource = gridCreatorSource.create(source);
        PartitionData gridTarget = gridCreatorTarget.create(target);
        for (uint32_t p = 0 ; p<gridTarget.size() ; ++p) {
            const Image& sourcePart = gridSource.at(data.items[p].i)->image();
            Image& targetPart = gridTarget.at(p)->image();
            Transform t = Transform(data.items[p].score.transform);
            t.copy(sourcePart, targetPart, data.items[p].score.contrast, data.items[p].score.brightness);
        }
    }
private:
    Image& _target;
    const int _iterations;
    const double _rmsEpsilon;
};

}

#endif // ENCODER_H
