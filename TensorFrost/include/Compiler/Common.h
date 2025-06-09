#pragma once

#include <vector>
#include <string>
#include <variant>
#include <unordered_map>
#include <stdexcept>
#include <memory>
#include <array>
#include <set>
#include <sstream>
#include <functional>
#include <stack>

namespace TensorFrost {
extern "C" {
    enum TFType {
        Float,
        Uint,
        Int,
        Bool,
        Tuple,
        None,
    };

    struct TFDataFormat {
        TFType type;
        size_t size;

        bool operator==(const TFDataFormat& other) const;
        bool operator!=(const TFDataFormat& other) const;
        int GetHash() const;
        bool operator<(const TFDataFormat& other) const;
        bool operator>(const TFDataFormat& other) const;
    };

#define TFTypeNone TFDataFormat{TFType::None, 0}
#define TFTypeTuple TFDataFormat{TFType::Tuple, 0}
#define TFTypeBool32 TFDataFormat{TFType::Bool, 32}
#define TFTypeFloat32 TFDataFormat{TFType::Float, 32}
#define TFTypeInt32 TFDataFormat{TFType::Int, 32}
#define TFTypeUint32 TFDataFormat{TFType::Uint, 32}
}

// Utility class to automatically resize and set elements in a vector
template<typename T>
class auto_vector : public std::vector<T> {
public:
    void set_element(size_t index, T&& value) {
        if (index >= this->size()) {
            this->resize(index + 1);
        }
        (*this)[index] = std::forward<T>(value);
    }
};

template<typename T, typename H = std::hash<T>>
struct VecHash {
    size_t operator()(const std::vector<T>& v) const noexcept {
        size_t h = 0;
        H hasher;
        for (const auto& x : v)
            h ^= hasher(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

enum class ArgType {
    Input,
    Index,
    Memory,
    Count,
};

inline std::string ToString(ArgType type) {
    switch (type) {
        case ArgType::Input: return "Input";
        case ArgType::Index: return "Index";
        case ArgType::Memory: return "Memory";
        default: return "Unknown";
    }
}

inline std::string ToString(const TFDataFormat& format) {
    switch (format.type) {
        case TFType::Float: return "float" + std::to_string(format.size);
        case TFType::Uint: return "uint" + std::to_string(format.size);
        case TFType::Int: return "int" + std::to_string(format.size);
        case TFType::Bool: return "bool" + std::to_string(format.size);
        case TFType::Tuple: return "tuple";
        case TFType::None: return "none";
        default: return "unknown";
    }
}

using uint = unsigned int;

struct Op;
struct Arguments;
struct OpBlock;
class OpBlockIterator;
struct ArgumentManager;
struct Argument;

using Attribute = std::variant<int, uint, float, bool, std::string>;
using AttributeMap = std::unordered_map<std::string, Attribute>;

//ostringstream conversion for Attribute
inline std::ostream& operator<<(std::ostream& os, const Attribute& attr) {
    std::visit([&os](const auto& v) { os << v; }, attr);
    return os;
}

inline std::string ToString(const Attribute& attr) {
    std::ostringstream oss;
    oss << attr;
    return oss.str();
}

}
namespace std {
template<>
struct hash<TensorFrost::TFDataFormat> {
    size_t operator()(const TensorFrost::TFDataFormat& f) const noexcept {
        return static_cast<size_t>(f.GetHash());
    }
};
}
