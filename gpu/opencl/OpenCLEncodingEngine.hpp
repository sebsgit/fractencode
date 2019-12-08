#pragma once

#include "encode/EncodingEngine2.hpp"

namespace Frac2 {
class OpenCLEncodingEngine : public AbstractEncodingEngine2 {
public:
    OpenCLEncodingEngine(const encode_parameters_t& params,
        const ImagePlane& sourceImage,
        const UniformGrid& sourceGrid);

    ~OpenCLEncodingEngine() override;

protected:
    encode_item_t encode_impl(const UniformGridItem& targetItem) const override;
    void init() override;
    void finalize() override;
private:
    struct Priv;
    std::unique_ptr<Priv> _d;
};
}
