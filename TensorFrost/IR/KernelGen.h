#pragma once

#include <utility>

#include "Backend/TensorMemory.h"
#include "Tensor/Tensor.h"

namespace TensorFrost {

bool IsBoundary(const Node* input, const Node* output, int arg_index,
                ArgType arg_type);

class Kernel {
 public:
	KernelIndexingMode indexing_mode_;
	Node* root;
	map<Node*, int> variables;
	map<Node*, int> memory;
	ArgMap shape;
	int dim = 0;

	string kernel_name_;
	string generated_code_;
};

class Program {
 public:
	IR* ir_;
	vector<Kernel> kernels_;
	function<void()> unload_callback;
	string generated_code_;
	string program_name = "TensorProgram";

	function<main_func> execute_callback;

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