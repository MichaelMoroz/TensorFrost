#include "../include/Common.h"

using namespace TensorFrost;

bool TFDataFormat::operator==(const TFDataFormat &other) const {
    return type == other.type && size == other.size;
}

bool TFDataFormat::operator!=(const TFDataFormat &other) const {
    return !(*this == other);
}

int TFDataFormat::GetHash() const {
    return (int)type << 16 | (int)size;
}

bool TFDataFormat::operator<(const TFDataFormat &other) const {
    return GetHash() < other.GetHash();
}

bool TFDataFormat::operator>(const TFDataFormat &other) const {
    return GetHash() > other.GetHash();
}

