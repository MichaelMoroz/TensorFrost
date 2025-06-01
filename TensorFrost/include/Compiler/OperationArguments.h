#pragma once
#include "Common.h"

namespace TensorFrost {

struct Argument {
    ArgType type;
    Op* from = nullptr;
    Op* to = nullptr;
    int index = 0;
};

struct Arguments {
    Op* parent_op = nullptr;
    auto_vector<std::unique_ptr<Argument>> inputs;
    std::set<std::pair<int, Argument*>> used_at;

    void AddInput(ArgType type, Op* from, int index = 0);
    bool CheckValidity(bool throw_error = false) const;
};

struct ShapeArgs : Arguments {
    std::vector<int> TryGetShape(int default_value = 256) const;
    float GetSizeEstimate();
    void ExpandDimensionsTo(int new_dim);

    bool CompareShape(const ShapeArgs& other, bool throw_error = false) const {
        //TODO: Implement shape comparison logic
    }
};

struct ArgumentManager {
    Op* parent_op = nullptr;
    std::array<std::unique_ptr<Arguments>, (int)ArgType::Count> type_args;

    ArgumentManager(Op* parent);
    void AddArgument(Op* from, ArgType type, int index = 0);
    void SetAsOutput(Argument *arg);
    void SetArguments(ArgType type, std::vector<Op*> args);
};

}