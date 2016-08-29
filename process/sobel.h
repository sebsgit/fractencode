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

}

#endif // SOBEL_H
