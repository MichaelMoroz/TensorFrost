#pragma once
#include "Common.h"

namespace TensorFrost {

struct Argument {
    Op* from = nullptr;
    Op* to = nullptr;
    int index = 0;
};

struct ArgumentManager {
    Op* parent_op = nullptr;
    auto_vector<std::unique_ptr<Argument>> inputs;
    std::set<std::pair<int, Argument*>> used_at;

    ArgumentManager(Op* parent);
    void AddArgument(Op &from, int index = 0);
    void SetAsOutput(Argument *arg);
    void RemoveOutput(Argument *arg);
    void SetArguments(std::vector<Op *> args);
    void Remove(int index);
    void RemoveAll();

    std::vector<Argument*> Args() const;
    std::vector<Op*> Inputs() const;
};

}