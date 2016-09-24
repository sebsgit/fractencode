#ifndef ABSTRACTPROCESSOR_H
#define ABSTRACTPROCESSOR_H

#include "image/image.h"

namespace Frac {

class AbstractProcessor {
public:
	virtual ~AbstractProcessor() {}
	virtual Image process(const Image& image) const = 0;
};

}

#endif // ABSTRACTPROCESSOR_H
