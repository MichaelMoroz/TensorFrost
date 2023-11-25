#pragma once

#include <utility>

#include "Backend/TensorMemory.h"
#include "Tensor/Tensor.h"

namespace TensorFrost {

bool IsBoundary(const Node* input, const Node* output, int arg_index,
                Argument::Type arg_type);

enum KernelType {
	Memory,
	Compute,
};

class Kernel {
 public:
	KernelType type_;
	Node* begin_;
	vector<Node*> variables;
	vector<Node*> memory;
	function<void(TensorMemoryManager*, vector<uint>, vector<uint>, uint)>
	    execute_callback;
};

class Program {
 public:
	IR* ir_;
	vector<Kernel> kernels_;
	function<void()> unload_callback;

	explicit Program(IR* ir) : ir_(ir) {}

	void AddKernel(KernelType type, Node* begin, vector<Node*> variables,
	               vector<Node*> memory) {
		kernels_.push_back({type, begin, std::move(variables), std::move(memory)});
	}
};

Program* GenerateProgram(IR* ir);

}  // namespace TensorFrost