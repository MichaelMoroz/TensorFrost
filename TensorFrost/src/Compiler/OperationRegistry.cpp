#include "Compiler/Operation.h"

using namespace std;

namespace TensorFrost {
OpSpec::OpSpec(std::string op_name, OverloadsMap overloads_list, int block_count) {
    name = std::move(op_name);
    overloads = std::move(overloads_list);
    blocks = block_count;
}

TFDataFormat OpSpec::GetOutputType(const std::vector<TFDataFormat> &args) const {
    auto it = overloads.find(args);
    if (it == overloads.end()) {
        throw std::runtime_error("No overload found for operation: " + name + " with args: " + to_string(args.size()));
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

OverloadsMap ovr(const std::string& input) {
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

vector<OpSpec> default_operations = {
    OpSpec("const", ovr("f(); u(); i(); b(); tuple()")),

    OpSpec("add", ovr("f(f,f); u(u,u); i(i,i)")),
    OpSpec("sub", ovr("f(f,f); u(u,u); i(i,i)")),
    OpSpec("mul", ovr("f(f,f); u(u,u); i(i,i)")),
    OpSpec("div", ovr("f(f,f); u(u,u); i(i,i)")),

    OpSpec("vmap", ovr("tuple()"), 1),
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