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

    void Compile() {
        StartExecutionContext(&context);
        //constant folding
        ApplyOpTransform(*GetBaseBlock(), [](Op& op) {
            OpSpec* spec = GetOpSpec(op.opcode);
            if(spec->constant_fold) { // Specification has a constant folding function
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
            }
        });
        EndExecutionContext();
    }

    std::string DebugPrint() const;
};
}