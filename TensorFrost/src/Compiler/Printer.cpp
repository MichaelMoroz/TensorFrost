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
        return VariableName(arg->from) + (arg->from->output_count > 1 ? "[" + std::to_string(arg->from_index) + "]" : "");
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

std::string PrintShape(const Shape& shape) {
    std::vector<std::string> dims;
    for (const auto& dim : shape.dimensions) {
        dims.push_back(VariableName(dim.op));
    }
    return PrintArray(dims, "[", "]", ", ");
}

std::string PrintAttribute(Attribute attr) {
    std::ostringstream oss;
    std::visit([&oss](const auto& v) { oss << v; }, attr);
    return oss.str();
}

std::string PrintOp(const Op* op) {
    std::ostringstream os;
    os << ToString(op->type) << " " << op->varname;
    if (op->opcode == "const") {
        //return "";
        os << " = " << op->attributes.at("value");
    } else {
        std::string inputs = PrintArguments(op->args->inputs, "", "");
        std::vector<std::string> attributes;
        for (const auto& [key, value] : op->attributes) {
            attributes.push_back(key + ": " + PrintAttribute(value));
        }
        std::string attributes_str = PrintArray(attributes, "{", "}");

        std::string shape_str = "";// PrintShape(ComputeShape(Value(op)));

        os << shape_str << (op->output_count > 1 ? "[" + std::to_string(op->output_count) + "]" : "");
        os << " = " << op->opcode << "(" << PrintArray({inputs, attributes_str}) << ")";
    }
    return os.str();
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

std::string PrintBlock(OpBlock &root) {
    return IterateWithLocalState<std::string>(root, [](OpBlock::Iterator &it, std::string& current, const std::vector<std::string> &kids) {
        std::string result = PrintOp(*it);
        if(result.empty()) return;
        if (!kids.empty()) {
            std::vector<std::string> indented;
            indented.reserve(kids.size());
            for (auto &s : kids) indented.push_back(AddIndent(s, 4));
            result += PrintArray(indented, " { \n", "}", "} else { \n");
        }
        result += '\n';
        current += result;
    });
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
