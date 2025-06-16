#include "Compiler/Operation.h"

namespace TensorFrost {
Op::Op(std::string op_name): opcode(std::move(op_name)) {
    args = std::make_unique<ArgumentManager>(this);
    type = TFNone;
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

Attribute Op::GetAttribute(const std::string &name) const {
    auto it = attributes.find(name);
    if (it == attributes.end()) {
        throw std::runtime_error("Attribute '" + name + "' not found in operation '" + opcode + "'");
    }
    return it->second;
}

bool Op::Compare(const Op &other) const {
    bool both_const = (opcode == "const" && other.opcode == "const");
    if (both_const) {
        // Compare constant values directly
        Attribute this_value = GetAttribute("value");
        Attribute other_value = other.GetAttribute("value");
        return (this_value == other_value);
    }
    return false; // TODO: Implement more complex comparison logic for non-constant operations
}

void ApplyOpTransform(OpBlock &block, const std::function<void(Op &)> &transform) {
    for (auto& op : block.ops) {
        for (auto& sub_block : op->blocks) {
            ApplyOpTransform(*sub_block, transform);
        }
        transform(*op);
    }
}

void IterateOver(OpBlock &block, const std::function<void(OpBlock::Iterator&)> &transform) {
    for (OpBlock::Iterator it = block.begin(); it.valid(); it.next()) {
        for (auto& sub_block : it->blocks) {
            IterateOver(*sub_block, transform);
        }
        transform(it);
    }
}

void ReverseIterateOver(OpBlock &block, const std::function<void(OpBlock::Iterator&)> &transform) {
    for (OpBlock::Iterator it = block.end(); it.valid(); it.prev()) {
        for (auto& sub_block : it->blocks) {
            ReverseIterateOver(*sub_block, transform);
        }
        transform(it);
    }
}

std::set<Op*> CollectDependencies(std::vector<Op*> ops) {
    std::set<Op*> dependencies;
    std::function<void(Op*)> collect_dependencies = [&](Op* op) {
        if (op == nullptr || dependencies.contains(op)) return; // Already processed
        dependencies.insert(op);
        for (auto& input : op->args->Get(ArgType::Input)->inputs) {
            collect_dependencies(input->from);
        }
        for (auto& input : op->args->Get(ArgType::Index)->inputs) {
            collect_dependencies(input->from);
        }
        collect_dependencies(op->parent_block->parent_op); // Parent depends on this operation
    };
    for (Op* op : ops) {
        collect_dependencies(op);
    }
    return dependencies;
}
}
