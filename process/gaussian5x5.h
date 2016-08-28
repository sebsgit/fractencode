#ifndef GAUSSIAN5X5_H
#define GAUSSIAN5X5_H

#include "image/image.h"

namespace Frac {

class AbstractProcessor {
public:
    virtual Image process(const Image& image) const = 0;
};

class GaussianBlur5x5 : public AbstractProcessor {
public:
    Image process(const Image& image) const override;
};
}

#endif // GAUSSIAN5X5_H
