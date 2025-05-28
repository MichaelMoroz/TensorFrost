#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <variant>

namespace TensorFrost {

struct OpSpec {
    std::string name;
    int arity = -1;                // -1 variadic
};

inline std::unordered_map<std::string, OpSpec>& registry() {
    static std::unordered_map<std::string, OpSpec> r;
    return r;
}

inline void reg(OpSpec s) { registry()[s.name] = std::move(s); }

} // namespace ir