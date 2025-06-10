#include "Compiler/TFProgram.h"

namespace TensorFrost {
TFProgram::TFProgram(std::function<std::pair<std::vector<Op*>, std::vector<Op*>>()> program_fn) {
    StartExecutionContext(&context);

    auto [ins, outs] = program_fn();
    program_inputs = std::move(ins);
    program_outputs = std::move(outs);
    if (program_outputs.empty()) {
        throw std::runtime_error("Program must have at least one output");
    }

    AssignVariableNames(*GetBaseBlock());
    EndExecutionContext();
}

std::string TFProgram::DebugPrint() const {
    std::string program_header = "TFProgram(inputs=" + PrintArray(TransformVector(program_inputs, VariableName), "[", "]") + ") {\n";
    std::string inner_code = PrintBlock(*context.base_block);
    inner_code += "return " + PrintArray(TransformVector(program_outputs, VariableName), "[", "]") + ";\n";
    return program_header + AddIndent(inner_code, 2) + "}\n";
}
}
