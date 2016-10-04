#include "image/image.h"
#include "encode/encoder.h"
#include "transform.h"
#include "metrics.h"
#include "partition.h"
#include "partition/quadtreepartition.h"
#include "utils/timer.h"
#include "process/gaussian5x5.h"
#include "process/sobel.h"
#include <iostream>
#include <cassert>
#include <cstring>

class CmdArgs {
public:
	std::string inputPath;
	Frac::Encoder::encode_parameters_t encoderParams;
	int decodeSteps = -1;
	double decodeRms = 0.00001;
	bool saveDecodeSteps = false;
	bool color = false;
	bool useQuadtree = false;
	bool preSample = false;

	CmdArgs(int argc, char** argv) {
		assert(argc > 1);
		this->inputPath = argv[1];
		this->_parse(argv + 2, argc - 2);
	}
	explicit CmdArgs(const std::string& path)
	:inputPath(path)
 	{
	}
private:
	void _parse(char** s, const int count) {
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
			} else {
				std::cout << "unrecognized parameter: " << tmp << '\n';
				exit(0);
			}
			++index;
		}
		if (encoderParams.targetGridSize >= encoderParams.sourceGridSize || encoderParams.targetGridSize < 2 || encoderParams.sourceGridSize < 2) {
			std::cout << "invalid source / target size\n";
			throw std::exception();
		}
	}
};

static void test_partition() {
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

static Frac::Image encode_image(const CmdArgs& args, Frac::Image image) {
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
	Encoder encoder(image, args.encoderParams, *sourceCreator, *targetCreator);
	std::cout << "encoded in " << timer.elapsed() << " s.\n";
	auto data = encoder.data();
	uint32_t w = image.width(), h = image.height();
	AbstractBufferPtr<Image::Pixel> buffer = Buffer<Image::Pixel>::alloc(w * h);
	buffer->memset(0);
	Image result = Image(buffer, w, h, w);
	timer.start();
	Decoder decoder(result, args.decodeSteps, args.decodeRms, args.saveDecodeSteps);
	auto stats = decoder.decode(data);
	std::cout << "decoded in " << timer.elapsed() << " s.\n";
	std::cout << "decode stats: " << stats.iterations << " steps, rms: " << stats.rms << "\n";
	return result;
}

static void test_encoder(const CmdArgs& args) {
	using namespace Frac;
	Timer timer;
	timer.start();
	if (args.color == false) {
		Image image(args.inputPath.c_str());
		Image result = encode_image(args, image);
		result.savePng("result.png");
	} else {
		PlanarImage image(args.inputPath.c_str());
		Image y = encode_image(args, image.y());
		Image u = encode_image(args, image.u());
		Image v = encode_image(args, image.v());
		PlanarImage result(y, u, v);
		result.savePng("result.png");
	}
	std::cout << "total time: " << timer.elapsed() << " s.\n";
}

static void test_statistics() {
	using namespace Frac;
	int w = 64;
	int h = 64;
	double imageSum = 0.0;
	uint32_t stride = w + 64;
	auto buffer = Buffer<Image::Pixel>::alloc(stride * h);
	for (int i=0 ; i<h ; ++i)
		for (int j=0 ; j<w ; ++j) {
			buffer->get()[j + stride*i] = i + j;
			imageSum += i + j;
		}
	const double mean = imageSum / (w * h);
	double variance = 0.0;
	for (int i=0 ; i<h ; ++i)
		for (int j=0 ; j<w ; ++j) {
			const auto p = buffer->get()[j + stride*i];
			variance += (p - mean) * (p - mean);
		}
	variance /= (w * h);
	Image image(buffer, w, h, stride);
	const double testSum = ImageStatistics::sum(image);
	if (fabs(testSum - imageSum) > 0.001) {
		std::cout << "expected sum " << imageSum << ", actual " << testSum << '\n';
		exit(0);
	}
	const double testVariance = ImageStatistics::variance(image);
	if (fabs(testVariance - variance) > 0.001) {
		std::cout << "expected variance " << variance << ", actual " << testVariance << '\n';
		exit(0);
	}
}

static void test_sobel() {
	using namespace Frac;
	const float expected_x[] = {
		4,8,8,8,8,8,-15,-15,8,8,8,8,8,8,8,4,
		4,8,8,8,8,8,-38,-38,8,8,8,8,8,-15,-15,4,
		4,8,8,8,8,8,-15,-15,8,8,8,8,8,-38,-38,4,
		4,8,8,8,-15,-15,8,8,8,8,8,8,8,-15,-15,4,
		4,8,8,8,-38,-38,8,8,8,8,8,-15,-15,8,8,4,
		4,8,8,8,-15,-15,8,8,8,8,8,-38,-38,8,8,4,
		4,8,-15,-15,8,8,8,8,8,8,8,-15,-15,8,8,4,
		4,8,-38,-38,8,8,8,8,8,-15,-15,8,8,8,8,4,
		4,8,-15,-15,8,8,8,8,8,-38,-38,8,8,8,8,4,
		-19,-15,8,8,8,8,8,8,8,-15,-15,8,8,8,8,4,
		-42,-38,8,8,8,8,8,-15,-15,8,8,8,8,8,8,4,
		-19,-15,8,8,8,8,8,-38,-38,8,8,8,8,8,-15,-19,
		4,8,8,8,8,8,8,-15,-15,8,8,8,8,8,-38,-42,
		4,8,8,8,8,-15,-15,8,8,8,8,8,8,8,-15,-19,
		4,8,8,8,8,-38,-38,8,8,8,8,8,-15,-15,8,4,
		4,8,8,8,8,-15,-15,8,8,8,8,8,-61,-61,8,4
	};

	const float expected_y[] = {
		64,64,64,64,64,64,41,-5,-28,-28,-28,-28,-28,-28,-28,-28,
		36,36,36,36,36,36,36,36,36,36,36,36,36,13,-33,-56,
		-56,-56,-56,-56,-56,-56,-33,13,36,36,36,36,36,36,36,36,
		36,36,36,36,13,-33,-56,-56,-56,-56,-56,-56,-56,-33,13,36,
		36,36,36,36,36,36,36,36,36,36,36,13,-33,-56,-56,-56,
		-56,-56,-56,-56,-33,13,36,36,36,36,36,36,36,36,36,36,
		36,36,13,-33,-56,-56,-56,-56,-56,-56,-56,-33,13,36,36,36,
		36,36,36,36,36,36,36,36,36,13,-33,-56,-56,-56,-56,-56,
		-56,-56,-33,13,36,36,36,36,36,36,36,36,36,36,36,36,
		13,-33,-56,-56,-56,-56,-56,-56,-56,-33,13,36,36,36,36,36,
		36,36,36,36,36,36,36,13,-33,-56,-56,-56,-56,-56,-56,-56,
		-33,13,36,36,36,36,36,36,36,36,36,36,36,36,13,-33,
		-56,-56,-56,-56,-56,-56,-56,-33,13,36,36,36,36,36,36,36,
		36,36,36,36,36,13,-33,-56,-56,-56,-56,-56,-56,-56,-33,13,
		36,36,36,36,36,36,36,36,36,36,36,36,13,-33,-56,-56,
		-28,-28,-28,-28,-28,-5,41,64,64,64,64,64,41,-5,-28,-28,
	};

	const int w = 16;
	const int h = 16;
	const uint32_t stride = w + 64;
	auto buffer = Buffer<Image::Pixel>::alloc(stride * h);
	for (int i=0 ; i<h ; ++i)
		for (int j=0 ; j<w ; ++j) {
			buffer->get()[j + stride*i] = (i * w + j) % 23;
		}
	Image image(buffer, w, h, stride);
	auto result = SobelOperator().calculate(image);
	for (size_t i=0 ; i<result->size() ; ++i) {
		if (result->get()[i].dx != expected_x[i]) {
			std::cout << "error in sobel operator (x): " << i << ':' << result->get()[i].dx  << '/' << expected_x[i] << '\n';
		}
		if (result->get()[i].dy != expected_y[i]) {
			std::cout << "error in sobel operator (y): " << i << ':' << result->get()[i].dy  << '/' << expected_y[i] << '\n';
		}
	}
}

static void test_sampler() {
	using namespace Frac;
	const Size32u size(8, 8);
	const size_t stride = 64;
	Image image(size.x(), size.y(), stride);
	for (uint32_t y=0 ; y<size.y() ; ++y) {
		for (uint32_t x=0 ; x<size.x() ; ++x) {
			image.data()->get()[x + y * stride] = 4;
		}
	}
	SamplerBilinear sampler(image);
	Transform t;
	while (1) {
		for (uint32_t y=0 ; y<size.y() ; ++y) {
			for (uint32_t x=0 ; x<size.x() ; ++x) {
				if(sampler(x, y, t, size) != 4) {
					std::cout << "sampler error\n";
					exit(0);
				}
			}
		}
		if(t.next() == Transform::Id)
			break;
	}
}

int main(int argc, char *argv[])
{
	test_statistics();
	test_partition();
	test_sobel();
	test_sampler();

	if (argc > 1) {
		Frac::Image image(argv[1]);
		if (image.data()) {

			Frac::Image test = Frac::SobelOperator().process(image);
			test.savePng("sobel.png");
			test = Frac::NonMaximumSuppressionOperator().edgeImage(Frac::SobelOperator().calculate(image), image.width(), image.height());
			test.savePng("sobel_nonmax.png");

			test_encoder(CmdArgs(argc, argv));
		}
	}
	else {
		test_encoder(CmdArgs("small.png"));
	}
	return 0;
}
