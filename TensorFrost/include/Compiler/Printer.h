#pragma once
#include "Operation.h"
#include "OperationBlocks.h"

namespace TensorFrost {

void PrintOp(const Op& op, std::ostream& os);
std::string PrintBlock(OpBlock& base_block);
void AssignVariableNames(OpBlock &block);

}
