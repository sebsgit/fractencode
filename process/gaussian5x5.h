#ifndef GAUSSIAN5X5_H
#define GAUSSIAN5X5_H

#include "process/abstractprocessor.h"

namespace Frac {

class GaussianBlur5x5 : public AbstractProcessor {
public:
    Image process(const Image& image) const override;
};
}

#endif // GAUSSIAN5X5_H
