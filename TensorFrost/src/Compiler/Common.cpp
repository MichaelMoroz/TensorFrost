#include "Compiler/Common.h"

namespace TensorFrost {
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

TFDataFormat GetTypeFromAttribute(const Attribute& attr) {
    if (std::holds_alternative<int>(attr)) {
        return TFTypeInt32;
    } else if (std::holds_alternative<uint>(attr)) {
        return TFTypeUint32;
    } else if (std::holds_alternative<float>(attr)) {
        return TFTypeFloat32;
    } else if (std::holds_alternative<bool>(attr)) {
        return TFTypeBool32;
    }
    throw std::runtime_error("Unsupported attribute type for TFDataFormat conversion");
}
}
