#pragma once

#include "Tensor/Tensor.h"
#include "Backend/TensorMemory.h"

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
    vector<Node*> variables;
    vector<Node*> memory;
	function<void(TensorMemoryManager*, vector<uint>, vector<uint>, uint)> execute_callback;
};

class Program {
public:
	IR* ir_;
	vector<Kernel> kernels_;
	function<void()> unload_callback;

    Program(IR* ir) : ir_(ir) {}

	void AddKernel(KernelType type, Node* begin, vector<Node*> variables, vector<Node*> memory) {
		kernels_.push_back({ type, begin, variables, memory });
	}
};

Program* GenerateProgram(IR* ir);
    
}   // namespace TensorFrost