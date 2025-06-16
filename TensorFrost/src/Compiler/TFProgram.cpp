#include "Compiler/TFProgram.h"

namespace TensorFrost {
TFProgram::TFProgram(std::function<std::pair<std::vector<Value>, std::vector<Value>>()> program_fn) {
    StartExecutionContext(&context);

    auto [ins, outs] = program_fn();
    program_inputs = std::move(ins);
    program_outputs = std::move(outs);
    if (program_outputs.empty()) {
        throw std::runtime_error("Program must have at least one output");
    }
    EndExecutionContext();
}

void TFProgram::Compile() {
    StartExecutionContext(&context);
    ConstantFold();
    RemoveUnused();
    AssignVariableNames(*GetBaseBlock());
    EndExecutionContext();
}

void TFProgram::ConstantFold() {
    ApplyOpTransform(*GetBaseBlock(), [](Op& op) {
       OpSpec* spec = GetOpSpec(op.opcode);
       if(!spec->const_fold) return; // Skip if no constant folding is defined for this operation
       AttributeVector inputs;
       for (const auto& arg : op.args->Get(ArgType::Input)->inputs) {
           if(!arg->from->attributes.contains("value")) {
               return; // Skip if some argument does not have a constant value
           }
           inputs.push_back(arg->from->attributes.at("value"));
       }
       Attribute result = spec->const_fold(inputs);
       op.attributes["value"] = result; // Set the result as a constant value
       op.opcode = "const"; // Change the opcode to constant
       op.args->RemoveAll(); // Clear all arguments
   });
}

void TFProgram::RemoveUnused() {
     std::set<Op*> used_ops = CollectDependencies(values_to_ops(program_outputs));
     IterateOver(*GetBaseBlock(), [&](OpBlock::Iterator& it) {
         if (!used_ops.contains(*it)) {
             it.remove(); // Remove unused operations
         }
     });
}

// Converts multilevel vmap operations into a sequence of vmaps with concatenated shape
void TFProgram::CombineVmapDepthwise() {
    IterateOver(*GetBaseBlock(), [&](OpBlock::Iterator& it) {
         static OpBlock* last_vmap_block = nullptr;
         static OpBlock* current_vmap_block = nullptr;
         static Shape current_shape;
     });
}

std::string TFProgram::DebugPrint() const {
    std::string program_header = "TFProgram(inputs=" + PrintArray(TransformVector(values_to_ops(program_inputs), VariableName), "[", "]") + ") {\n";
    std::string inner_code = PrintBlock(*context.base_block);
    inner_code += "return " + PrintArray(TransformVector(values_to_ops(program_outputs), VariableName), "[", "]") + ";\n";
    return program_header + AddIndent(inner_code, 2) + "}\n";
}
}
