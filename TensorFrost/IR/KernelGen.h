#pragma once

#include <utility>

#include "Backend/TensorMemory.h"
#include "Tensor/Tensor.h"

namespace TensorFrost {

bool IsBoundary(const Node* input, const Node* output, int arg_index,
                Arg::Type arg_type);

enum KernelType {
	Memory,
	Compute,
};

class Kernel {
 public:
	KernelType type_;
	KernelIndexingMode indexing_mode_;
	Node* begin_;
	map<Node*, int> variables;
	map<Node*, int> memory;
	ArgMap shape;
	int dim = 0;
	function<void(TensorMemoryManager*, vector<uint>, vector<uint>, vector<uint>)>
	    execute_callback;

	string generated_code_;
};

class Program {
 public:
	IR* ir_;
	vector<Kernel> kernels_;
	function<void()> unload_callback;
	string generated_code_;

	explicit Program(IR* ir) : ir_(ir) {}

	void AddKernel(KernelType type, KernelIndexingMode indexing_mode, Node* begin, map<Node*, int> variables, map<Node*, int> memory,
	               ArgMap shape, int dim) 
	{
		if (type == KernelType::Compute && dim == 0) {
			throw std::runtime_error("Invalid dimension");
		}
		kernels_.push_back(
		    {type, indexing_mode, begin, std::move(variables), std::move(memory), std::move(shape), dim});
	}
};

Program* GenerateProgram(IR* ir);

}  // namespace TensorFrost