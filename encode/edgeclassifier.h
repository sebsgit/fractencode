#ifndef EDGECLASSIFIER_H
#define EDGECLASSIFIER_H

#include "encode/classifier.h"
#include <memory>

namespace Frac {

class EdgeClassifier : public ImageClassifier {
    class Data;
public:
    EdgeClassifier(Image image);
    ~EdgeClassifier();
    bool compare(const PartitionItemPtr& targetItem, const PartitionItemPtr& sourceItem) const override;
    bool compare(const Image& a, const Image& b) const override;
private:
    std::unique_ptr<Data> _data;
};

}

#endif // EDGECLASSIFIER_H
