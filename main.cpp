#include "encode/Encoder2.hpp"
#include "encode/Quantizer.hpp"
#include "encode/encoder.h"
#include "image/Image2.hpp"
#include "image/image.h"
#include "image/metrics.h"
#include "image/partition.h"
#include "image/partition/quadtreepartition.h"
#include "process/gaussian5x5.h"
#include "process/sobel.h"
#include "image/transform.h"
#include "utils/timer.h"

#include "encode/EncodingEngine2.hpp"
#include "image/Image2.hpp"
#include "image/partition2.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>

#ifdef FRAC_TESTS
#define CATCH_CONFIG_MAIN
#endif
#include "tests/catch.hpp"

std::ostream& operator<<(std::ostream& out, const Frac::Size32u& p)
{
    out << p.x() << ',' << p.y() << ' ';
    return out;
}

class CmdArgs {
public:
    std::string inputPath;
    Frac::encode_parameters_t encoderParams;
    int decodeSteps = -1;
    double decodeRms = 0.00001;
    bool saveDecodeSteps = false;
    bool color = false;
    bool useQuadtree = false;
    bool preSample = false;
    bool logProgress = false;
    bool useOldImplementation = false;

    CmdArgs(int argc, char** argv)
    {
        assert(argc > 1);
        this->inputPath = argv[1];
        this->_parse(argv + 2, argc - 2);
    }
    explicit CmdArgs(const std::string& path)
        : inputPath(path)
    {
    }

private:
    void _parse(char** s, const int count)
    {
        int index = 0;
        while (index < count) {
            std::string tmp(s[index]);
            if (tmp == "--decode") {
                decodeSteps = atoi(s[index + 1]);
                ++index;
            } else if (tmp == "--source") {
                encoderParams.sourceGridSize = atoi(s[index + 1]);
                ++index;
            } else if (tmp == "--target") {
                encoderParams.targetGridSize = atoi(s[index + 1]);
                ++index;
            } else if (tmp == "--rms") {
                encoderParams.rmsThreshold = atof(s[index + 1]);
                ++index;
            } else if (tmp == "--smax") {
                encoderParams.sMax = atof(s[index + 1]);
                ++index;
            } else if (tmp == "--color") {
                color = true;
            } else if (tmp == "--quadtree") {
                useQuadtree = true;
            } else if (tmp == "--presample") {
                preSample = true;
            } else if (tmp == "--debug_decode") {
                saveDecodeSteps = true;
            } else if (tmp == "--nogpu") {
                encoderParams.nogpu = true;
            } else if (tmp == "--nocpu") {
                encoderParams.nocpu = true;
            } else if (tmp == "--noclassifier") {
                encoderParams.noclassifier = true;
            } else if (tmp == "--log") {
                logProgress = true;
            } else if (tmp == "--old") {
                useOldImplementation = true;
            } else {
                std::cout << "unrecognized parameter: " << tmp << '\n';
                exit(0);
            }
            ++index;
        }
        if (encoderParams.nogpu && encoderParams.nocpu) {
            std::cout << "invalid configuration: nocpu & nogpu!\n";
            throw std::exception();
        }
        if (encoderParams.targetGridSize >= encoderParams.sourceGridSize || encoderParams.targetGridSize < 2 || encoderParams.sourceGridSize < 2) {
            std::cout << "invalid source / target size\n";
            throw std::exception();
        }
    }
};

static void test_partition()
{
    using namespace Frac;
    const uint32_t w = 128, h = 128;
    const uint32_t gridSize = 8;
    AbstractBufferPtr<Image::Pixel> buffer = Buffer<Image::Pixel>::alloc(w * h);
    Image image = Image(buffer, w, h, w);
    GridPartitionCreator gridCreator(Size32u(gridSize, gridSize), Size32u(gridSize, gridSize));
    PartitionPtr grid = gridCreator.create(image);
    assert(grid->size() == (w * h) / (gridSize * gridSize));
    uint8_t color = 0;
    for (auto it : *grid) {
        Painter p(it->image());
        p.fill(color++);
    }
    //image.savePng("grid.png");
}

class StdoutReporter : public Frac::ProgressReporter {
public:
    void log(size_t done, size_t total) override
    {
        auto now = std::chrono::system_clock::now();
        std::chrono::duration<double> seconds = now - _lastPrintout;
        if (seconds.count() > 0.3) {
            _lastPrintout = now;
            const double percent = (100.0 * done) / total;
            std::stringstream ss;
            ss << percent;
            const auto toPrint = ss.str();
            rewind();
            this->_lastPrintLength = toPrint.size();
            std::cout << toPrint;
        }
    }

private:
    void rewind()
    {
        while (_lastPrintLength > 0) {
            --_lastPrintLength;
            std::cout << '\b';
        }
        std::fflush(stdout);
    }

private:
    size_t _lastPrintLength = 0;
    std::chrono::system_clock::time_point _lastPrintout;
};

static void encode_data_statistics(const Frac::grid_encode_data_t& data)
{
    using namespace Frac;
    double max_contrast = -1;
    double max_brightness = -1;
    double min_contrast = std::numeric_limits<double>::max();
    double min_brightness = std::numeric_limits<double>::max();
    for (const auto& d : data.encoded) {
        max_contrast = std::max(max_contrast, d.match.score.contrast);
        min_contrast = std::min(min_contrast, d.match.score.contrast);
        max_brightness = std::max(max_brightness, d.match.score.brightness);
        min_brightness = std::min(min_brightness, d.match.score.brightness);
    }
    // grid stats
    const int contrast_bits = 5;
    const int brighntess_bits = 7;
    std::cout << "----\n";
    std::cout << "grid element count: " << data.encoded.size() << "\n";
    std::cout << "contrast: " << min_contrast << ':' << max_contrast << '\n';
    std::cout << "brightness: " << min_brightness << ':' << max_brightness << '\n';
    // quantization stats
    Quantizerd quantBrightness(min_brightness, max_brightness, brighntess_bits);
    Quantizerd quantContrast(min_contrast, max_contrast, contrast_bits);
    std::map<int, int> brightness_buckets;
    std::map<int, int> contrast_buckets;
    for (const auto& d : data.encoded) {
        int bucket = quantContrast.quantized(d.match.score.contrast);
        ++contrast_buckets[bucket];

        bucket = quantBrightness.quantized(d.match.score.brightness);
        ++brightness_buckets[bucket];
    }
    std::cout << "contrast / brighntess quantization: " << contrast_buckets.size() << ' ' << brightness_buckets.size() << '\n';
    std::cout << "----\n";
}

static Frac::Image encode_image(const CmdArgs& args, Frac::Image image)
{
    using namespace Frac;
    const Size32u gridSize(args.encoderParams.targetGridSize, args.encoderParams.targetGridSize);
    const Size32u gridSizeSource(args.encoderParams.sourceGridSize, args.encoderParams.sourceGridSize);
    const Size32u gridOffset = gridSizeSource / args.encoderParams.latticeSize;
    std::unique_ptr<PartitionCreator> targetCreator;
    std::unique_ptr<PartitionCreator> sourceCreator;
    if (args.preSample) {
        sourceCreator.reset(new PreSampledPartitionCreator(gridSizeSource, gridOffset));
    } else {
        sourceCreator.reset(new GridPartitionCreator(gridSizeSource, gridOffset));
    }
    if (args.useQuadtree)
        targetCreator.reset(new QuadtreePartitionCreator(Size32u(args.encoderParams.sourceGridSize, args.encoderParams.sourceGridSize) / 2, gridSize));
    else
        targetCreator.reset(new GridPartitionCreator(gridSize, gridSize));
    Timer timer;
    timer.start();
    ProgressReporter* reporter = nullptr;
    if (args.logProgress)
        reporter = new StdoutReporter();
    Encoder encoder(image, args.encoderParams, *sourceCreator, *targetCreator, reporter);
    auto data = encoder.data();
    std::cout << "encoded in " << timer.elapsed() << " s.\n";
    std::cout << data.encoded.size() << " elements.\n";
    uint32_t w = image.width(), h = image.height();
    AbstractBufferPtr<Image::Pixel> buffer = Buffer<Image::Pixel>::alloc(w * h);
    buffer->memset(0);
    Image result = Image(buffer, w, h, w);
    timer.start();
    Decoder decoder(result, args.decodeSteps, args.decodeRms, args.saveDecodeSteps);
    auto stats = decoder.decode(data);
    std::cout << "decoded in " << timer.elapsed() << " s.\n";
    std::cout << "decode stats: " << stats.iterations << " steps, rms: " << stats.rms << "\n";
    encode_data_statistics(data);
    return result;
}

static Frac2::ImagePlane encode_image2(const CmdArgs& args, const Frac2::ImagePlane& image)
{
    using namespace Frac2;
    const Size32u gridSizeTarget(args.encoderParams.targetGridSize, args.encoderParams.targetGridSize);
    const Size32u gridSizeSource(args.encoderParams.sourceGridSize, args.encoderParams.sourceGridSize);
    const Size32u gridOffset = gridSizeSource / args.encoderParams.latticeSize;
    ProgressReporter2* reporter = nullptr;
    if (args.logProgress)
        reporter = new StdoutReporter2();

    auto sourceGrid = Frac2::createUniformGrid(image.size(), gridSizeSource, gridOffset);
    auto targetGrid = Frac2::createUniformGrid(image.size(), gridSizeTarget, gridSizeTarget);

    Timer timer;
    timer.start();
    Encoder2 encoder(image, args.encoderParams, sourceGrid, targetGrid, reporter);
    auto data = encoder.data();
    std::cout << "encoded in " << timer.elapsed() << " s.\n";
    std::cout << data.encoded.size() << " elements.\n";
    uint32_t w = image.width(), h = image.height();
    std::vector<uint8_t> imgData(w * h);
    std::memset(imgData.data(), 0, w * h);
    Frac2::ImagePlane result({ w, h }, w, std::move(imgData));
    timer.start();
    Decoder2 decoder(result, args.decodeSteps, args.decodeRms, args.saveDecodeSteps);
    auto stats = decoder.decode(data);
    std::cout << "decoded in " << timer.elapsed() << " s.\n";
    std::cout << "decode stats: " << stats.iterations << " steps, rms: " << stats.rms << "\n";
    encode_data_statistics(data);
    return result;
}

static void test_encoder(const CmdArgs& args)
{
    using namespace Frac;
    Timer timer;
    timer.start();
    if (args.color == false) {

        if (args.useOldImplementation) {
            Image image(args.inputPath.c_str());
            Image result = encode_image(args, image);
            result.savePng("result.png");
        } else {
            std::array<Frac2::ImagePlane, 3> image = Frac2::ImageIO::loadImage(args.inputPath);
            auto result = encode_image2(args, image[0]);
            Frac2::ImageIO::saveImage(result, "result.png");
        }
    } else {
        if (args.useOldImplementation) {
            PlanarImage image(args.inputPath.c_str());
            Image y = encode_image(args, image.y());
            Image u = encode_image(args, image.u());
            Image v = encode_image(args, image.v());
            PlanarImage result(y, u, v);
            result.savePng("result.png");
        } else {
            std::array<Frac2::ImagePlane, 3> image = Frac2::ImageIO::loadImage(args.inputPath);
            std::array<Frac2::ImagePlane, 3> planes {
                encode_image2(args, image[0]),
                encode_image2(args, image[1]),
                encode_image2(args, image[2])
            };
            Frac2::Image2<3> result(std::move(planes));
            Frac2::ImageIO::saveImage<3>(result, "result.png");
        }
    }
    std::cout << "total time: " << timer.elapsed() << " s.\n";
}

#ifndef FRAC_TESTS
int main(int argc, char* argv[])
{
	try {
		if (argc > 1) {
			Frac::Image image(argv[1]);
			if (image.data()) {
				test_encoder(CmdArgs(argc, argv));
			}
		} else {
			test_encoder(CmdArgs("small.png"));
		}
	} catch (const std::exception& exc) {
		std::cout << "EXCEPTION CAUGHT: " << exc.what() << '\n';
	}
    return 0;
}
#endif
