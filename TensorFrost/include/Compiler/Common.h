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
#include <map>

namespace TensorFrost {
extern "C" {
    enum TFType {
        Float,
        Uint,
        Int,
        Bool,
        None,
        Unknown,
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

#define TFNone TFDataFormat{TFType::None, 0}
#define TFUnknown TFDataFormat{TFType::Unknown, 0}
#define TFBool TFDataFormat{TFType::Bool, 32}
#define TFFloat32 TFDataFormat{TFType::Float, 32}
#define TFInt32 TFDataFormat{TFType::Int, 32}
#define TFUint32 TFDataFormat{TFType::Uint, 32}
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

inline std::string ToString(const TFDataFormat& format) {
    switch (format.type) {
        case TFType::Float: return "float" + std::to_string(format.size);
        case TFType::Uint: return "uint" + std::to_string(format.size);
        case TFType::Int: return "int" + std::to_string(format.size);
        case TFType::Bool: return "bool" + std::to_string(format.size);
        case TFType::None: return "void";
        default: return "unknown";
    }
}

using uint = unsigned int;

struct Op;
struct OpBlock;
class OpBlockIterator;
struct ArgumentManager;
struct Argument;
class Value;
struct Shape;

using Attribute = std::variant<int, uint, float, bool>;
using AttributeMap = std::unordered_map<std::string, Attribute>;
using AttributeVector = std::vector<Attribute>;
using Values = std::vector<Value>;

TFDataFormat GetTypeFromAttribute(const Attribute& attr);

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

template<typename Container, typename Func>
auto TransformVector(const Container& input, Func func) {
    using T2 = decltype(func(*std::begin(input)));
    std::vector<T2> output;
    output.reserve(input.size());
    for (const auto& item : input) {
        output.push_back(func(item));
    }
    return output;
}

template<typename T>
auto ConcatVectors(const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<T> result;
    result.reserve(a.size() + b.size());
    result.insert(result.end(), a.begin(), a.end());
    result.insert(result.end(), b.begin(), b.end());
    return result;
}

template<typename T>
auto SliceVector(const std::vector<T>& vec, size_t start, size_t end = -1) {
    if (end == -1 || end > vec.size()) {
        end = vec.size();
    }
    if (start >= end || start >= vec.size()) {
        return std::vector<T>();
    }
    return std::vector<T>(vec.begin() + start, vec.begin() + end);
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
