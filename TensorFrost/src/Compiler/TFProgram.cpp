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

void TFProgram::Compile() {
    StartExecutionContext(&context);
    ConstantFold();
    EndExecutionContext();
}

void TFProgram::ConstantFold() {
    ApplyOpTransform(*GetBaseBlock(), [](Op& op) {
       OpSpec* spec = GetOpSpec(op.opcode);
       if(!spec->constant_fold) return; // Skip if no constant folding is defined for this operation
       AttributeVector inputs;
       for (const auto& arg : op.args->Get(ArgType::Input)->inputs) {
           if(!arg->from->attributes.contains("value")) {
               return; // Skip if some argument does not have a constant value
           }
           inputs.push_back(arg->from->attributes.at("value"));
       }
       Attribute result = spec->constant_fold(inputs);
       op.attributes["value"] = result; // Set the result as a constant value
       op.opcode = "const"; // Change the opcode to constant
       op.args->RemoveAll(); // Clear all arguments
   });
}

void FindUsedOps(std::set<Op*>& used_ops, OpBlock& block) {
    for (auto& op : block.ops) {
        if (op->used_at.empty()) continue; // Skip unused operations
        used_ops.insert(op.get());
        for (auto& sub_block : op->blocks) {
            FindUsedOps(used_ops, *sub_block);
        }
    }
}

void TFProgram::RemoveUnused() {
     StartExecutionContext(&context);

     ApplyOpTransform(*GetBaseBlock(), [](Op& op) {
          // if (op.opcode == "const") return; // Skip constants
          // if (op.args->Get(ArgType::Input)->inputs.empty()) return; // Skip operations with no inputs
          // if (op.args->Get(ArgType::Output)->inputs.empty()) return; // Skip operations with no outputs
          // if (op.used_at.empty()) return; // Skip unused operations
     });
     EndExecutionContext();
}

std::string TFProgram::DebugPrint() const {
    std::string program_header = "TFProgram(inputs=" + PrintArray(TransformVector(program_inputs, VariableName), "[", "]") + ") {\n";
    std::string inner_code = PrintBlock(*context.base_block);
    inner_code += "return " + PrintArray(TransformVector(program_outputs, VariableName), "[", "]") + ";\n";
    return program_header + AddIndent(inner_code, 2) + "}\n";
}
}
