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

    Op(std::string op_name);
    Op(int value);
    Op(uint value);
    Op(float value);
    Op(bool value);
};


} // namespace ir

