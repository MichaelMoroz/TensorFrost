#pragma once
#include "Operation.h"
#include "ExecutionContext.h"
#include "Printer.h"

namespace TensorFrost {
class TFProgram {
public:
    ExecutionContext context;
    std::vector<Op*> program_inputs;
    std::vector<Op*> program_outputs;

    TFProgram(std::function<std::pair<std::vector<Op*>, std::vector<Op*>>()> program_fn);
    std::string DebugPrint() const;
};
}