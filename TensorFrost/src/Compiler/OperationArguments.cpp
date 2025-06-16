#include "Compiler/Operation.h"

namespace TensorFrost {

ArgumentManager::ArgumentManager(Op *parent): parent_op(parent) {
}

void ArgumentManager::AddArgument(Op &from, int index) {
    inputs.set_element(index, std::make_unique<Argument>(Argument{&from, parent_op, index}));
    from.args->SetAsOutput(inputs[index].get());
}

void ArgumentManager::SetAsOutput(Argument *arg) {
    used_at.insert({arg->index, arg});
}

void ArgumentManager::RemoveOutput(Argument *arg) {
    used_at.erase({arg->index, arg});
}

void ArgumentManager::SetArguments(std::vector<Op*> args) {
    for (size_t i = 0; i < args.size(); ++i) {
        AddArgument(*args[i], (int)i);
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

std::vector<Argument *> ArgumentManager::Args() const {
    std::vector<Argument*> result;
    for (const auto& arg : inputs) {
        if (arg) {
            result.push_back(arg.get());
        }
    }
    return result;
}

std::vector<Op *> ArgumentManager::Inputs() const {
    std::vector<Op*> result;
    for (const auto& arg : inputs) {
        if (arg && arg->from) {
            result.push_back(arg->from);
        }
    }
    return result;
}
}
