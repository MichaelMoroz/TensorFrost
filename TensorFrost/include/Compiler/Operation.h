#pragma once

#include "Common.h"
#include "OperationArguments.h"
#include "OperationBlocks.h"
#include "OperationRegistry.h"
#include "ExecutionContext.h"
#include "Overloads.h"

namespace TensorFrost {

struct Op {
    std::string opcode;
    std::unique_ptr<ArgumentManager> args;
    AttributeMap attributes;
    TFDataFormat type;
    std::vector<std::unique_ptr<OpBlock>> blocks;

    OpBlock* parent_block = nullptr;
    size_t index = 0; //might not be up to date
    std::string varname;

    Op(std::string op_name);
    OpBlock* NewBlock();
    OpBlock& GetBlock(int index = 0);

    void AddAttribute(const std::string& name, const Attribute& value);
    void ChangeAttribute(const std::string& name, const Attribute& value);

    Attribute GetAttribute(const std::string &name) const;

    bool Compare(const Op& other) const;
};

void ApplyOpTransform(OpBlock& block, const std::function<void(Op&)>& transform);
void IterateOver(OpBlock &block, const std::function<void(OpBlock::Iterator&)> &transform);
std::set<Op*> CollectDependencies(std::vector<Op*> ops);

template<class State, class Fn>
State IterateWithLocalState(OpBlock &block, Fn &&f) {
    State current = {};
    for (auto it = block.begin(); it.valid(); it.next()) {
        std::vector<State> kids;
        for (auto &sb : it->blocks)
            kids.push_back(IterateWithLocalState<State>(*sb, f));
        f(it, current, kids);
    }
    return current;
}

template<class State, class Fn>
void IterateWithGlobalState(OpBlock &block, State& current, Fn &&f) {
    for (auto it = block.begin(); it.valid(); it.next()) {
        std::vector<State> kids;
        for (auto &sb : it->blocks)
            kids.push_back(IterateWithGlobalState<State>(*sb,current, f));
        f(it, current, kids);
    }
}

} // namespace ir

