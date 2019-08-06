#pragma once

#include "encode/EncodingEngine2.hpp"
#include "image/ImageIO.hpp"
#include "encode/Classifier2.hpp"
#include "encode/DecodeUtils.hpp"

namespace Frac2 {
    class DummyReporter2 : public ProgressReporter2
    {
    public:
        void log(size_t, size_t) override {}
    };

    class Encoder2 {
    public:
        struct encode_stats_t {
            uint64_t rejectedMappings = 0;
            uint64_t totalMappings = 0;

            void print() {
                std::cout << "classifier rejected " << rejectedMappings << " out of " << totalMappings << " comparisons (" << (100.0*rejectedMappings) / totalMappings << ")%\n";
            }
        };

    public:
        Encoder2(const ImagePlane& image, const encode_parameters_t& p, const UniformGrid& sourcePartition, const UniformGrid& targetPartition, ProgressReporter2* reporter = nullptr)
            : _encodeParameters(p)
            , _reporter(reporter ? reporter : new DummyReporter2())
        {
            std::unique_ptr<Classifier2> classifier = std::make_unique<BrightnessBlocksClassifier2>(image, image);
            if (p.noclassifier)
                classifier = std::make_unique<DummyClassifier>(image, image);
            this->_estimator.reset(new TransformEstimator2(image, image, std::move(classifier), std::make_shared<TransformMatcher>(p.rmsThreshold, p.sMax), sourcePartition));
            this->_engine.reset(new EncodingEngineCore2(_encodeParameters, image, sourcePartition, *_estimator, _reporter.get()));
            this->_engine->encode(targetPartition);
            this->_stats.totalMappings = sourcePartition.items().size() * targetPartition.items().size();
        }
        grid_encode_data_t data() const {
            this->_stats.rejectedMappings = this->_estimator->rejectedMappings();
            this->_stats.print();
            return this->_engine->result();
        }
    private:
        const encode_parameters_t _encodeParameters;
        mutable encode_stats_t _stats;
        std::unique_ptr<TransformEstimator2> _estimator;
        std::unique_ptr<EncodingEngineCore2> _engine;
        std::unique_ptr<ProgressReporter2> _reporter;
    };

    class Decoder2 {
    public:
        struct decode_stats_t {
            int iterations;
            double rms;
        };
        Decoder2(ImagePlane& target, const int nMaxIterations = -1, const double rmsEpsilon = 0.00001, bool saveDecodeSteps = false)
            :_target(target)
            , _iterations(nMaxIterations < 0 ? 300 : nMaxIterations)
            , _rmsEpsilon(rmsEpsilon)
            , _saveSteps(saveDecodeSteps)
        {
        }
        decode_stats_t decode(const grid_encode_data_t& data) {
            std::vector<uint8_t> buffer(_target.height() * _target.stride());
            std::memset(buffer.data(), 100, buffer.size());
            ImagePlane source(_target.size(), _target.stride(), std::move(buffer));
            RootMeanSquare<TransformType::Id> metric;
            int i = 0;
            double rms = 0.0;
            if (_saveSteps)
                ImageIO::saveImage(source, "decode_debug0.png");
            for (; i < _iterations; ++i) {
                this->decodeStep(source, _target, data);
                if (_saveSteps) {
                    std::stringstream ss;
                    ss << i + 1;
                    ImageIO::saveImage(_target, "decode_debug" + ss.str() + ".png");
                }
                rms = metric.distance(source, _target);
                if (rms < _rmsEpsilon)
                    break;
                source = _target.copy();
            }
            return { i, rms };
        }
    private:
        void decodeStep(const ImagePlane& source, ImagePlane& target, const grid_encode_data_t& data) const {
            for (uint32_t p = 0; p < data.encoded.size(); ++p) {
                const encode_item_t enc = data.encoded.at(p);
                const item_match_t match = enc.match;
                GridItemBase sourcePatch{Point2du(match.x, match.y), match.sourceItemSize};
                GridItemBase targetPatch{Point2du(enc.x, enc.y), Size32u(enc.w, enc.h)};
                Frac::copy(source, target, sourcePatch, targetPatch, match.score.contrast, match.score.brightness, match.score.transform);
            }
        }
    private:
        ImagePlane& _target;
        const int _iterations;
        const double _rmsEpsilon;
        const bool _saveSteps = false;
    };
}
