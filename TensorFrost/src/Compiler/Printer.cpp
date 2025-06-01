#include "Compiler/Operation.h"
#include "Compiler/ExecutionContext.h"
#include "Compiler/Printer.h"

using namespace std;

namespace TensorFrost {

std::string VariableName(const Op* op) {
    if (op->opcode == "const") {
        return ToString(op->attributes.at("value"));
    }
    return op->varname;
}

bool PrintArguments(const auto_vector<std::unique_ptr<Argument>>& vec, std::ostringstream &os, string begin, string end) {
    if (vec.empty()) return false;
    os << begin;
    bool first = true;
    for (const auto& v : vec) {
        if (!first) os << ", ";
        first = false;
        os << VariableName(v->from);
    }
    os << end;
    return true;
}

void PrintOp(const Op* op, std::ostringstream &os) {
    os << ToString(op->type) << " " << op->varname;
    PrintArguments(op->args->Get(ArgType::Shape)->inputs, os, "[", "]");

    if (op->opcode == "const") {
        os << " = " << op->attributes.at("value");
        return;
    } else {
        os << " = " << op->opcode << "(";
        // Print inputs
        bool printed = PrintArguments(op->args->Get(ArgType::Input)->inputs, os, "", "");
        printed |= PrintArguments(op->args->Get(ArgType::Index)->inputs, os, ", index={", "}");
        printed |= PrintArguments(op->args->Get(ArgType::Memory)->inputs, os, ", memory={", "}");
        if (!op->attributes.empty()) {
            if (printed) os << ", ";
            os <<"{";
            bool first = true;
            for (const auto& [key, value] : op->attributes) {
                if (!first) os << ", ";
                first = false;
                os << key << ": ";
                std::visit([&os](const auto& v) { os << v; }, value);
            }
            os << "}";
        }
        os <<")";
    }
}

std::string AddIndent(const std::string& str, int indent) {
    // Add indentation to each line of the string
    std::string indented;
    std::istringstream iss(str);
    std::string line;
    while (std::getline(iss, line)) {
        indented += std::string(indent, ' ') + line + "\n";
    }
    return indented;
}

std::string PrintBlock(OpBlock &block) {
    auto oss = std::ostringstream();
    for (auto it = block.begin(); it.valid(); it.next()) {
        PrintOp(*it, oss);
        if(it->blocks.size() > 0) {
            bool first = true;
            oss << " {\n";
            for (auto& sub_block : it->blocks) {
                if (!first) oss << "{\n";
                oss<<AddIndent(PrintBlock(*sub_block.get()), 4);
                first = false;
                oss << "}\n";
            }
        }
        oss << "\n";
    }
    return oss.str();
}

void AssignVariableNames(OpBlock &block) {
    ApplyOpTransform(block, [](Op &op) {
        static size_t var_counter = 0;
        size_t index = var_counter++;
        op.varname = "var" + std::to_string(index);
        op.index = index;
    });
}

}
