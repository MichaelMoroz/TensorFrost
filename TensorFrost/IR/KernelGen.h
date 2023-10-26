#pragma once

#include "Tensor/Tensor.h"

namespace TensorFrost {

bool IsBoundary(const Node* input, const Node* output, int arg_index, Argument::Type arg_type);
    
}   // namespace TensorFrost