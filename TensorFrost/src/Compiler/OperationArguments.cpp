#include "Compiler/Operation.h"

namespace TensorFrost {
void Arguments::AddInput(ArgType type, Op *from, int index) {
    inputs.set_element(index, std::make_unique<Argument>(Argument{type, from, parent_op, index}));
    from->args->SetAsOutput(inputs[index].get());
}

void Arguments::RemoveInput(int index) {
    if (index < 0 || index >= inputs.size()) return;
    Argument *arg = inputs[index].get();
    if(!arg || !arg->from) throw std::runtime_error("Invalid argument");
    arg->from->args->RemoveOutput(arg);
    inputs[index].reset();
}

bool Arguments::CheckValidity(bool throw_error) const {
    for (const auto& input : inputs) {
        if (!input || !input->from) {
            if (throw_error) {
                throw std::runtime_error("Invalid argument");
            }
            return false;
        }
    }
    return true;
}

std::vector<Argument *> Arguments::Args() const {
    std::vector<Argument*> result;
    for (const auto& arg : inputs) {
        if (arg) {
            result.push_back(arg.get());
        }
    }
    return result;
}

std::vector<Op *> Arguments::Inputs() const {
    std::vector<Op*> result;
    for (const auto& arg : inputs) {
        if (arg && arg->from) {
            result.push_back(arg->from);
        }
    }
    return result;
}

bool ShapeArgs::CompareShape(const ShapeArgs &other, bool throw_error) const {
    //TODO: Implement shape comparison logic
    return true; // Placeholder
}

ArgumentManager::ArgumentManager(Op *parent): parent_op(parent) {
    for (int i = 0; i < (int)ArgType::Shape; ++i) {
        type_args[i] = std::make_unique<Arguments>();
        type_args[i]->parent_op = parent;
    }
    type_args[(int)ArgType::Shape] = std::make_unique<ShapeArgs>();
    type_args[(int)ArgType::Shape]->parent_op = parent;
}

void ArgumentManager::AddArgument(Op *from, ArgType type, int index) {
    type_args[(int)type]->AddInput(type, from, index);
}

void ArgumentManager::SetAsOutput(Argument *arg) {
    type_args[(int)arg->type]->used_at.insert({arg->index, arg});
}

void ArgumentManager::RemoveOutput(Argument *arg) {
    type_args[(int)arg->type]->used_at.erase({arg->index, arg});
}

void ArgumentManager::SetArguments(ArgType type, std::vector<Op*> args) {
    for (size_t i = 0; i < args.size(); ++i) {
        AddArgument(args[i], type, (int)i);
    }
}

Arguments* ArgumentManager::Get(ArgType type) const {
    return type_args[(int)type].get();
}

Arguments * ArgumentManager::operator[](ArgType type) const {
    return Get(type);
}
}
