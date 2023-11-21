#pragma once

#include "Tensor/Tensor.h"

namespace TensorFrost {

bool IsBoundary(const Node* input, const Node* output, int arg_index, Argument::Type arg_type);

enum KernelType
{
    Memory,
    Compute,
};

class Kernel
{
public:
    KernelType type_;
    Node* begin_;
};

class Program {
public:
	IR* ir_;
	vector<Kernel> kernels_;
    Program(IR* ir) : ir_(ir) {}

	void AddKernel(KernelType type, Node* begin) {
		kernels_.push_back({ type, begin });
	}
};

Program* GenerateProgram(IR* ir);
    
}   // namespace TensorFrost