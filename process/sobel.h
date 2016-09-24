#ifndef SOBEL_H
#define SOBEL_H

#include "process/abstractprocessor.h"

namespace Frac {

class SobelOperator : public AbstractProcessor {
public:
	typedef struct {
		float dx = 0.0f;
		float dy = 0.0f;
	} result_t;
	Image process(const Image& image) const; // only for testing / debugging
	AbstractBufferPtr<result_t> calculate(const Image& image) const;
};

class NonMaximumSuppressionOperator {
public:
	// callback for edge pixel: (x, y, isStrongEdge)
	typedef std::function<void(uint32_t, uint32_t, bool)> pixel_callback_t;

	explicit NonMaximumSuppressionOperator(pixel_callback_t cb = [](uint32_t, uint32_t, bool){ }) : _pixelCallback(cb) {

	}
	Image edgeImage(AbstractBufferPtr<SobelOperator::result_t> gradients, uint32_t imageWidth, uint32_t imageHeight) const;
	void process(AbstractBufferPtr<SobelOperator::result_t> gradients, uint32_t imageWidth, uint32_t imageHeight) const;
private:
	const float _hiCutoff = 35.0f; // pixels with magnitude greater than this are part of "strong" edges
	const float _loCutoff = 23.0f; // pixels with magnitude greater that this are part of "weak" edges, lower are discarded
	mutable pixel_callback_t _pixelCallback;
};

}

#endif // SOBEL_H
