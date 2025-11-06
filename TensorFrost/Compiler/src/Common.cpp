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
        return TFInt32;
    } else if (std::holds_alternative<uint>(attr)) {
        return TFUint32;
    } else if (std::holds_alternative<float>(attr)) {
        return TFFloat32;
    } else if (std::holds_alternative<bool>(attr)) {
        return TFBool;
    }
    throw std::runtime_error("Unsupported attribute type for TFDataFormat conversion");
}
}
