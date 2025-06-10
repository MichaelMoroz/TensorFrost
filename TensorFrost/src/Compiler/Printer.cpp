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

std::vector<std::string> StringifyArguments(const auto_vector<std::unique_ptr<Argument>>& vec) {
    return TransformVector(vec, [](const std::unique_ptr<Argument>& arg) {
        return VariableName(arg->from);
    });
}

std::string PrintArray(std::vector<std::string> items, const std::string &begin, const std::string &end, const std::string& separator) {
    std::ostringstream oss;
    if (items.empty()) return "";
    oss << begin;
    bool first = true;
    for (const auto& item : items) {
        if (item.empty()) continue; // Skip empty items
        if (!first) oss << separator;
        first = false;
        oss << item;
    }
    oss << end;
    return oss.str();
}

std::string PrintArguments(const auto_vector<std::unique_ptr<Argument>>& vec, string begin, string end) {
    return PrintArray(StringifyArguments(vec), begin, end);
}

std::string PrintArguments(const Arguments* args) {
    if (!args) return "";
    std::vector<std::string> inputs = StringifyArguments(args->inputs);
    std::vector<std::string> outputs;
    for (const auto& arg : args->used_at) {
        if (arg.second->to) {
            outputs.push_back(VariableName(arg.second->to));
        }
    }
    std::string inputs_str = PrintArray(inputs, "inputs={", "}");
    std::string outputs_str = PrintArray(outputs, "outputs={", "}");
    return "[" + inputs_str + ", " + outputs_str + "]";
}

std::string PrintAttribute(Attribute attr) {
    std::ostringstream oss;
    std::visit([&oss](const auto& v) { oss << v; }, attr);
    return oss.str();
}

void PrintOp(const Op* op, std::ostringstream &os) {
    os << ToString(op->type) << " " << op->varname;
    if (op->opcode == "const") {
        os << " = " << op->attributes.at("value");
    } else {
        std::string inputs = PrintArguments(op->args->Get(ArgType::Input)->inputs, "", "");
        std::string index = PrintArguments(op->args->Get(ArgType::Index)->inputs, "index={", "}");
        // std::string inputs = "args=" + PrintArguments(op->args->Get(ArgType::Input));
        // std::string index = "index=" + PrintArguments(op->args->Get(ArgType::Index));
        std::vector<std::string> attributes;
        for (const auto& [key, value] : op->attributes) {
            attributes.push_back(key + ": " + PrintAttribute(value));
        }
        std::string attributes_str = PrintArray(attributes, "{", "}");

        os << " = " << op->opcode << "(" << PrintArray({inputs, index, attributes_str}) << ")";
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
            std::vector<std::string> blocks;
            for (auto& sub_block : it->blocks) {
                blocks.push_back(AddIndent(PrintBlock(*sub_block.get()), 4));
            }
            oss << PrintArray(blocks, " { \n", "}", "} { \n");
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
