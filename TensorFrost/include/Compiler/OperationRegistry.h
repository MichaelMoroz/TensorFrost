#pragma once
#include "Common.h"

namespace TensorFrost {

using OverloadsMap = std::unordered_map<std::vector<TFDataFormat>, TFDataFormat, VecHash<TFDataFormat>>;

enum class OpClass {
    Operator,
    UnaryOperator,
    Function,
    Copy,
    Keyword,
    Parallel,
    Variable,
    TypeCast,
    TypeReinterpret,
    Constant,
    TernaryOperator,
    Memory,
    None,
};

enum class OpProp {
    HasShape,
    Load,
    Store,
    MemoryOp,
    Set,
};

struct OpSpec {
    std::string name;
    OverloadsMap overloads;
    OpClass op_class = OpClass::None;
    std::set<OpProp> properties;
    int blocks = 0;

    OpSpec(std::string op_name, OverloadsMap overloads_list, int block_count = 0,
           OpClass op_class_type = OpClass::None, std::set<OpProp> props = {});

    TFDataFormat GetOutputType(const std::vector<TFDataFormat>& args) const;
};

void RegisterOperation(const OpSpec& spec);
OpSpec* GetOpSpec(const std::string& name);

}