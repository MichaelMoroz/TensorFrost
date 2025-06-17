#include "Compiler/Operation.h"

namespace TensorFrost {

Value Argument::From() const {
    return Value(from, from_index);
}

ArgumentManager::ArgumentManager(Op *parent): parent_op(parent) {
}

void ArgumentManager::AddArgument(Value from, int arg_index) {
    inputs.set_element(arg_index, std::make_unique<Argument>(Argument{from.op, parent_op, arg_index, from.out_index}));
    from.op->args->SetAsOutput(inputs[arg_index].get());
}

void ArgumentManager::SetAsOutput(Argument *arg) {
    used_at.insert({arg->arg_index, arg});
}

void ArgumentManager::RemoveOutput(Argument *arg) {
    used_at.erase({arg->arg_index, arg});
}

void ArgumentManager::SetArguments(Values args) {
    for (size_t i = 0; i < args.size(); ++i) {
        AddArgument(args[i], (int)i);
    }
}

void ArgumentManager::Remove(int index) {
    if (index < 0 || index >= inputs.size()) {
        throw std::out_of_range("Index out of range for arguments");
    }
    Argument *arg = inputs[index].get();
    if(!arg || !arg->from) throw std::runtime_error("Invalid argument");
    arg->from->args->RemoveOutput(arg);
    inputs[index].reset();
}

void ArgumentManager::RemoveAll() {
    for (size_t i = 0; i < inputs.size(); ++i) {
        Remove((int)i);
    }
    inputs.clear();
}

Values ArgumentManager::Inputs() const {
    Values result;
    for (const auto& arg : inputs) {
        if (arg) result.push_back(arg->From());
    }
    return result;
}
}
