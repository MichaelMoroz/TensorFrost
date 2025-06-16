#include "Compiler/Operation.h"

using namespace std;

namespace TensorFrost {
ArgSpec::ArgSpec(std::string io, std::map<char, std::set<TFDataFormat>> types,
    std::map<char, std::set<ArgProp>> props) {
    this->props = std::move(props);
    this->types = std::move(types);
    if (io.empty()) {
        throw std::invalid_argument("Argument specification cannot be empty");
    }
    // Parse argument specification (e.g., "x(x,y,t)" -> out = 'x', in = {'x', 'y', 't'}, or "z(y,z,...)" -> out = 'z', in = {'y', 'z'}, variadic = true)
    out = io[0];
    //get substring between parentheses
    size_t start = io.find('(');
    size_t end = io.find(')', start);
    //split the substring by commas
    std::stringstream ss(io.substr(start + 1, end - start - 1));
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token == "...") {
            variadic = true;
            break; // Variadic argument, stop parsing further
        } 
        if (token.empty()) continue; // Skip empty tokens
        in.push_back(token[0]);
    }
}

bool ArgSpec::IsValid(std::vector<TFDataFormat> inputs, TFDataFormat output) const {
    if (variadic) {
        if (in.empty() || inputs.empty()) return false;
    } else {
        if (inputs.size() != in.size()) return false;
    }

    auto name_of = [&](size_t i) -> const char& {
        return variadic ? in.front() : in[i];
    };

    std::unordered_map<char, TFDataFormat> seen;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& n = name_of(i);
        if (seen.count(n) && !(seen[n] == inputs[i]))
            throw std::runtime_error("Conflicting types for arg " + n);
        seen[n] = inputs[i];

        auto a = types.find(n);
        if (a != types.end() && !a->second.count(inputs[i]))
            return false;
    }

    auto ao = types.find(out);
    if (ao != types.end() && !ao->second.count(output)) return false;
    if (seen.count(out) && !(seen[out] == output)) return false;

    if (variadic) {
        for (size_t i = 1; i < inputs.size(); ++i)
            if (!(inputs[i] == inputs[0])) return false;
    }
    return true;
}

TFDataFormat ArgSpec::EstimateOutputType(const std::vector<TFDataFormat> &inputs) const {
    if (variadic && inputs.empty()) return TFUnknown;

    auto name_of = [&](size_t i) -> const char& {
        return variadic ? in.front() : in[i];
    };

    std::unordered_map<char, TFDataFormat> seen;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& n = name_of(i);
        if (seen.count(n) && !(seen[n] == inputs[i]))
            throw std::runtime_error("Conflicting types for arg " + n);
        seen[n] = inputs[i];
    }

    if (seen.count(out)) return seen[out];

    auto ao = types.find(out);
    if (ao != types.end() && ao->second.size() == 1)
        return *ao->second.begin();

    return TFUnknown;
}

std::vector<std::set<ArgProp>> ArgSpec::InputProperties(const std::vector<TFDataFormat> &inputs) const {
    if ((!variadic && inputs.size() != in.size()) ||
        (variadic && (in.empty() || inputs.empty())))
        return {};

    auto name_of = [&](size_t i) -> const char& {
        return variadic ? in.front() : in[i];
    };

    std::vector<std::set<ArgProp>> out;
    out.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& n = name_of(i);
        auto p = props.find(n);
        out.push_back(p == props.end() ? std::set<ArgProp>() : p->second);
    }
    return out;
}

TFDataFormat OpSpec::GetOutputType(const std::vector<TFDataFormat> &args) const {
    TFDataFormat ret = arg_spec.EstimateOutputType(args);
    return ret;
}

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

#define DEF_OP(op_name, overload_str, operation_class, ...) \
    OpSpec{ .name = op_name, .arg_spec = ArgSpec overload_str, .op_class = operation_class,  __VA_ARGS__ }

vector<OpSpec> default_operations = {
    DEF_OP("memory", ("x(y,...)"), OpClass::Memory, .props = {OpProp::HasShape}),
    DEF_OP("load", ("x(x,y,...)", {{'y', {TFInt32}}}, {{'x', {ArgProp::IgnoreShape}}}),
        OpClass::Function, .props = {OpProp::Load, OpProp::MemoryOp}),
    DEF_OP("store", ("x(x,x,y,...)", {{'y', {TFInt32}}}, {{'x', {ArgProp::IgnoreShape}}}),
        OpClass::Function, .props = {OpProp::Store, OpProp::MemoryOp}),

    DEF_OP("const", ("x()"), OpClass::Constant),
    DEF_OP("copy", ("x(x)"), OpClass::Copy),
    DEF_OP("add", ("x(x,x)"), OpClass::Operator,
        .const_fold = BIN_OP_FOLD(+)),
    DEF_OP("sub", ("x(x,x)"), OpClass::Operator,
        .const_fold = BIN_OP_FOLD(-)),
    DEF_OP("mul", ("x(x,x)"), OpClass::Operator,
        .const_fold = BIN_OP_FOLD(*)),
    DEF_OP("div", ("x(x,x)"), OpClass::Operator,
        .const_fold = BIN_OP_FOLD(/)),
    DEF_OP("sin", ("x(x)"), OpClass::UnaryOperator,
        .const_fold = UN_FUNC_FOLD(std::sinf)),
    DEF_OP("cos", ("x(x)"), OpClass::UnaryOperator,
        .const_fold = UN_FUNC_FOLD(std::cosf)),
    DEF_OP("tan", ("x(x)"), OpClass::UnaryOperator,
        .const_fold = UN_FUNC_FOLD(std::tanf)),

    DEF_OP("eq", ("x(y,y)", {{'x', {TFBool}}}), OpClass::Operator,
        .const_fold = BIN_FUNC_FOLD(std::equal_to<>())),
    DEF_OP("ne", ("x(y,y)", {{'x', {TFBool}}}), OpClass::Operator,
        .const_fold = BIN_FUNC_FOLD(std::not_equal_to<>())),
    DEF_OP("lt", ("x(y,y)", {{'x', {TFBool}}}), OpClass::Operator,
        .const_fold = BIN_FUNC_FOLD(std::less<>())),
    DEF_OP("le", ("x(y,y)", {{'x', {TFBool}}}), OpClass::Operator,
        .const_fold = BIN_FUNC_FOLD(std::less_equal<>())),
    DEF_OP("gt", ("x(y,y)", {{'x', {TFBool}}}), OpClass::Operator,
        .const_fold = BIN_FUNC_FOLD(std::greater<>())),
    DEF_OP("ge", ("x(y,y)", {{'x', {TFBool}}}), OpClass::Operator,
        .const_fold = BIN_FUNC_FOLD(std::greater_equal<>())),

    DEF_OP("tofloat", ("x(y)", {{'x', {TFFloat32}}}), OpClass::Function,
        .const_fold = UN_FUNC_FOLD(static_cast<float>)),
    DEF_OP("toint", ("x(y)", {{'x', {TFInt32}}}), OpClass::Function,
        .const_fold = UN_FUNC_FOLD(static_cast<int32_t>)),
    DEF_OP("touint", ("x(y)", {{'x', {TFUint32}}}), OpClass::Function,
        .const_fold = UN_FUNC_FOLD(static_cast<uint32_t>)),
    DEF_OP("tobool", ("x(y)", {{'x', {TFBool}}}), OpClass::Function,
        .const_fold = UN_FUNC_FOLD(static_cast<bool>)),

    DEF_OP("unpack", ("x(y)"), OpClass::Function),

    // Operations with blocks
    DEF_OP("vmap", ("x(y,...)", {{'x', {TFTuple}}, {'y', {TFInt32}}}), OpClass::Parallel, .props = {OpProp::HasShape}, .blocks = 1),
    DEF_OP("if_cond", ("x(y)", {{'x', {TFNone}}, {'y', {TFBool}}}), OpClass::Function, .blocks = 2),
    DEF_OP("loop", ("x(x,x,x)", {{'x', {TFInt32}}}), OpClass::Function, .blocks = 1),

    DEF_OP("phi", ("x(x,...)"), OpClass::Phi),
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