#pragma once

#include "Tensor/Tensor.h"

namespace TensorFrost {

bool IsBoundary(const Node* input, const Node* output, int arg_index, Argument::Type arg_type);

class Kernel
{
public:
    vector<Node*> inputs_;
    vector<Node*> outputs_;
    Node* begin_;
};

class Program {
    vector<Kernel> kernels_;
};
    
}   // namespace TensorFrost