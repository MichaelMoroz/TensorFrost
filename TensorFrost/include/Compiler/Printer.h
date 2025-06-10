#pragma once
#include "Operation.h"
#include "OperationBlocks.h"

namespace TensorFrost {

std::string VariableName(const Op* op);
void PrintOp(const Op& op, std::ostream& os);
std::string PrintBlock(OpBlock& base_block);
void AssignVariableNames(OpBlock &block);
std::string PrintAttribute(Attribute attr);
std::string AddIndent(const std::string& str, int indent);
std::string PrintArray(std::vector<std::string> items, const std::string& begin = "", const std::string& end = "",
                       const std::string& separator = ", ");

}
