#include "Compiler/Operation.h"

namespace TensorFrost {
Op::Op(std::string op_name): opcode(std::move(op_name)) {
    args = std::make_unique<ArgumentManager>(this);
    type = TFTypeNone;
}

OpBlock* Op::NewBlock() {
    blocks.emplace_back(std::make_unique<OpBlock>(this));
    return blocks.back().get();
}

OpBlock& Op::GetBlock(int index) {
    if (index < 0 || index >= blocks.size()) {
        throw std::out_of_range("Block index out of range");
    }
    return *blocks[index];
}

void Op::AddAttribute(const std::string &name, const Attribute &value) {
    if (attributes.find(name) != attributes.end()) {
        throw std::runtime_error("Attribute '" + name + "' already exists in operation '" + opcode + "'");
    }
    attributes[name] = value;
}

void Op::ChangeAttribute(const std::string &name, const Attribute &value) {
    if (attributes.find(name) == attributes.end()) {
        throw std::runtime_error("Attribute '" + name + "' not found in operation '" + opcode + "'");
    }
    attributes[name] = value;
}

void Op::GetAttribute(const std::string &name, Attribute &value) const {
    auto it = attributes.find(name);
    if (it == attributes.end()) {
        throw std::runtime_error("Attribute '" + name + "' not found in operation '" + opcode + "'");
    }
    value = it->second;
}
}
