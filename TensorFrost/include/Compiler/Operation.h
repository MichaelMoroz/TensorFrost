#pragma once

#include "Common.h"
#include "OperationArguments.h"
#include "OperationBlocks.h"
#include "OperationRegistry.h"
#include "ExecutionContext.h"
#include "Overloads.h"

namespace TensorFrost {

struct Op {
    OpBlock* parent_block = nullptr;

    size_t index = 0; //might not be up to date
    std::string opcode;
    std::string varname;
    std::unique_ptr<ArgumentManager> args;
    AttributeMap attributes;
    TFDataFormat type;
    std::vector<std::unique_ptr<OpBlock>> blocks;

    Op(std::string op_name);
    OpBlock* NewBlock();
    OpBlock& GetBlock(int index = 0);

    Op& operator[](int index);
    Op& operator[](std::vector<Op*> indices);

    void AddAttribute(const std::string& name, const Attribute& value);
    void ChangeAttribute(const std::string& name, const Attribute& value);
    void GetAttribute(const std::string& name, Attribute& value) const;
};


} // namespace ir

