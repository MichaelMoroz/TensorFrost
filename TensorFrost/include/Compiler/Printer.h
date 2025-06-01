#pragma once
#include "Operation.h"
#include "OperationBlocks.h"

namespace TensorFrost {

void PrintOp(const Op& op, std::ostream& os);
std::string PrintTree(const OpBlock& base_block);
}
