#include "Compiler/Operation.h"

using namespace std;

namespace TensorFrost {
TFDataFormat OpSpec::GetOutputType(const std::vector<TFDataFormat> &args) const {
    if (props.contains(OpProp::ShapeArgs) || args.empty()) {
        return overloads.find({})->second;
    }
    auto it = overloads.find(args);
    if (it == overloads.end()) {
        std::string error_msg = "No overload found for operation: " + name + " with args: (";
        for (const auto& arg : args) {
            error_msg += ToString(arg) + ", ";
        }
        if (!args.empty()) {
            error_msg.pop_back(); // Remove last comma
            error_msg.pop_back(); // Remove last space
        }
        error_msg += ")";

        throw std::runtime_error(error_msg);
    }
    return it->second;
}

static const std::unordered_map<std::string, TFDataFormat> tok = {
    {"f", TFTypeFloat32},
    {"i", TFTypeInt32},
    {"u", TFTypeUint32},
    {"tuple", TFTypeTuple},
    {"b", TFTypeBool32},
    {"void", TFTypeNone},
};

static std::string trim(std::string_view s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return std::string{s.substr(a, b - a)};
}

OverloadsMap GenerateOverloadMap(const std::string& input) {
    OverloadsMap out;
    std::stringstream ss(input);
    std::string stmt;
    while (std::getline(ss, stmt, ';')) {
        stmt = trim(stmt);
        if (stmt.empty()) continue;
        auto l = stmt.find('('), r = stmt.find(')');
        if (l == std::string::npos || r == std::string::npos || r < l) throw std::runtime_error("Overload syntax error: " + stmt);
        auto tgt = trim(stmt.substr(0, l));
        auto args = stmt.substr(l + 1, r - l - 1);
        std::vector<TFDataFormat> key;
        std::stringstream as(args);
        std::string tokarg;
        while (std::getline(as, tokarg, ',')) {
            tokarg = trim(tokarg);
            key.push_back(tok.at(tokarg));
        }
        out.emplace(std::move(key), tok.at(tgt));
    }
    return out;
}

#define DEF_OP(op_name, overload_str, operation_class, ...) \
    OpSpec{ .name = op_name, .overloads = GenerateOverloadMap(overload_str), .op_class = operation_class,  __VA_ARGS__ }

#define BIN_OP_FOLD(op) \
make_fold2([](auto a, auto b) { \
    if constexpr (std::is_same_v<decltype(a), bool> || std::is_same_v<decltype(b), bool>) { \
        return static_cast<int>(a) op static_cast<int>(b); \
    } else { \
        return a op b; \
    } \
})

#define UN_OP_FOLD(op) \
    make_fold1([](auto a) { return op a; })

#define UN_FUNC_FOLD(op) \
    make_fold1([](auto a) { return op(a); })

#define BIN_FUNC_FOLD(op) \
    make_fold2([](auto a, auto b) { return op(a, b); })

#define TERN_FUNC_FOLD(op) \
    make_fold3([](auto a, auto b, auto c) { return op(a, b, c); })

vector<OpSpec> default_operations = {
    DEF_OP("memory", "f(); u(); i(); b(); tuple()", OpClass::Memory, .props = {OpProp::ShapeArgs}),
    DEF_OP("load", "f(f); u(u); i(i); b(b)", OpClass::Function, .props = {OpProp::Load, OpProp::MemoryOp}),
    DEF_OP("store", "f(f); u(u); i(i); b(b)", OpClass::Function, .props = {OpProp::Store, OpProp::MemoryOp}),

    DEF_OP("const", "f(); u(); i(); b(); tuple()", OpClass::Constant),
    DEF_OP("copy", "f(f); u(u); i(i); b(b)", OpClass::Copy),
    DEF_OP("add", "f(f,f); u(u,u); i(i,i)", OpClass::Operator, .constant_fold = BIN_OP_FOLD(+)),
    DEF_OP("sub", "f(f,f); u(u,u); i(i,i)", OpClass::Operator, .constant_fold = BIN_OP_FOLD(-)),
    DEF_OP("mul", "f(f,f); u(u,u); i(i,i)", OpClass::Operator, .constant_fold = BIN_OP_FOLD(*)),
    DEF_OP("div", "f(f,f); u(u,u); i(i,i)", OpClass::Operator, .constant_fold = BIN_OP_FOLD(/)),
    DEF_OP("sin", "f(f); u(u); i(i)", OpClass::UnaryOperator, .constant_fold = UN_FUNC_FOLD(std::sinf)),
    DEF_OP("cos", "f(f); u(u); i(i)", OpClass::UnaryOperator, .constant_fold = UN_FUNC_FOLD(std::cosf)),
    DEF_OP("tan", "f(f); u(u); i(i)", OpClass::UnaryOperator, .constant_fold = UN_FUNC_FOLD(std::tanf)),


    DEF_OP("tofloat", "f(i); f(u); f(b)", OpClass::TypeCast),
    DEF_OP("toint", "i(f); i(u); i(b)", OpClass::TypeCast),
    DEF_OP("touint", "u(f); u(i); u(b)", OpClass::TypeCast),
    DEF_OP("tobool", "b(f); b(i); b(u)", OpClass::TypeCast),

    DEF_OP("unpack_tuple_int", "i(tuple)", OpClass::Function),

    DEF_OP("vmap", "tuple()", OpClass::Parallel, .props = {OpProp::ShapeArgs}, .blocks = 1),
};

std::unordered_map<string, unique_ptr<OpSpec>> CreateOperationRegistry() {
    std::unordered_map<string, unique_ptr<OpSpec>> registry;
    for (const auto& op : default_operations) {
        registry[op.name] = std::make_unique<OpSpec>(op);
    }
    return registry;
}

std::unordered_map<string, unique_ptr<OpSpec>> operation_registry = CreateOperationRegistry();

void TensorFrost::RegisterOperation(const OpSpec &spec) {
    if (operation_registry.contains(spec.name)) {
        throw std::runtime_error("Operation already registered: " + spec.name);
    }
    operation_registry[spec.name] = std::make_unique<OpSpec>(spec);
}

OpSpec* TensorFrost::GetOpSpec(const std::string &name) {
    if (!operation_registry.contains(name)) {
        throw std::runtime_error("Operation not found: " + name);
    }
    return operation_registry[name].get();
}
}