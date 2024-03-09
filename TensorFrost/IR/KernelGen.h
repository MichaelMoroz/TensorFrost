#pragma once

#include <utility>

#include "Backend/TensorMemory.h"
#include "Tensor/Tensor.h"

namespace TensorFrost {

bool IsBoundary(const Node* input, const Node* output, int arg_index,
                Arg::Type arg_type);

class Kernel {
 public:
	KernelIndexingMode indexing_mode_;
	Node* kernel_node_;
	map<Node*, int> variables;
	map<Node*, int> memory;
	ArgMap shape;
	int dim = 0;

	string generated_code_;
};

class Program {
 public:
	IR* ir_;
	vector<Kernel> kernels_;
	function<void()> unload_callback;
	string generated_code_;

	function<void(uint* in, uint* out, uint* mem,
	              uint(uint*&, uint*, uint dim),
	              void(uint))>
	    execute_callback;

	explicit Program(IR* ir) : ir_(ir) {}

	void AddKernel(KernelIndexingMode indexing_mode, Node* kernel_node, map<Node*, int> variables, map<Node*, int> memory,
	               ArgMap shape, int dim) 
	{
		kernels_.push_back(
		    {indexing_mode, kernel_node, std::move(variables), std::move(memory), std::move(shape), dim});
	}
};

Program* GenerateProgram(IR* ir);

string GetOperationListing(const IR&, bool compact = false,
                           map<Node*, string> invalid = {});

}  // namespace TensorFrost