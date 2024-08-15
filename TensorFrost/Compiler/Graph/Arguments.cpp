#include "Arguments.h"

namespace TensorFrost {

void ArgumentManager::ClearOutputs() {
    outputs_.clear();
}

const map<ArgType, string> arg_type_names = {
    {ArgType::Input, "Input"}, {ArgType::Index, "Index"}, {ArgType::Shape, "Shape"},
    {ArgType::Memory, "Memory"}, {ArgType::None, "None"},
};

string TypeToString(ArgType type) {
    return arg_type_names.at(type);
}


} // namespace TensorFrost