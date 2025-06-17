#pragma once
#include "Common.h"

namespace TensorFrost {

struct Argument {
    Op* from = nullptr;
    Op* to = nullptr;
    int arg_index = 0; // Index in to's arguments
    int from_index = 0; // Index of from's output

    Value From() const;
};

struct ArgumentManager {
    Op* parent_op = nullptr;
    auto_vector<std::unique_ptr<Argument>> inputs;
    std::set<std::pair<int, Argument*>> used_at;

    ArgumentManager(Op* parent);
    void AddArgument(Value from, int arg_index = 0);
    void SetAsOutput(Argument *arg);
    void RemoveOutput(Argument *arg);
    void SetArguments(Values args);
    void Remove(int index);
    void RemoveAll();

    Values Inputs() const;
};

}
