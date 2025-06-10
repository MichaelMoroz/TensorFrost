#pragma once
#include "Operation.h"
#include "ExecutionContext.h"
#include "Printer.h"

namespace TensorFrost {
class TFProgram {
public:
    ExecutionContext context;
    std::vector<Value> program_inputs;
    std::vector<Value> program_outputs;

    TFProgram(std::function<std::pair<std::vector<Value>, std::vector<Value>>()> program_fn);

    void Compile();
    void ConstantFold();
    void RemoveUnused();

    std::string DebugPrint() const;
};
}