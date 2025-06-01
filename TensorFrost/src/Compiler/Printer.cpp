#include "Compiler/Operation.h"
#include "Compiler/ExecutionContext.h"
#include "Compiler/Printer.h"

using namespace std;

namespace TensorFrost {
void PrintOp(const Op &op, std::ostringstream &os) {
    os << "Op: " << op.opcode << "\n";
    os << "Type: " << ToString(op.type) << "\n";
    os << "Arguments:\n";
    for (int i = 0; i < (int)ArgType::Count; ++i) {
        const auto args = op.args->GetArguments(static_cast<ArgType>(i));
        if (args) {
            os << "  " << ToString(static_cast<ArgType>(i)) << ":\n";
            for (const auto& arg : args->inputs) {
                if (arg) {
                    os << "    From: " << (arg->from ? arg->from->opcode : "nullptr")
                            << ", Index: " << arg->index << "\n";
                }
            }
        }
    }
    os << "Attributes:\n";
    for (const auto& [key, value] : op.attributes) {
        os << "  " << key << ": ";
        std::visit([&os](const auto& v) { os << v; }, value);
        os << "\n";
    }
}

std::string PrintTree(const OpBlock &base_block) {
    OpBlockIterator it(const_cast<OpBlock*>(&base_block));
    std::ostringstream oss;
    while (Op* op = it.next()) {
        PrintOp(*op, oss);
        oss << "\n";
    }
    return oss.str();
}

}
