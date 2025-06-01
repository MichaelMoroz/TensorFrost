#pragma once
#include "Common.h"

namespace TensorFrost {

using OverloadsMap = std::unordered_map<std::vector<TFDataFormat>, TFDataFormat, VecHash<TFDataFormat>>;

struct OpSpec {
    std::string name;
    OverloadsMap overloads;
    int blocks = 0;

    OpSpec(std::string op_name, OverloadsMap overloads_list, int block_count = 0);

    TFDataFormat GetOutputType(const std::vector<TFDataFormat>& args) const;
};

void RegisterOperation(const OpSpec& spec);
OpSpec* GetOpSpec(const std::string& name);

}