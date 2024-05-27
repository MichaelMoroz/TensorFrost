#pragma once

#include <utility>
#include <limits.h>
#include <float.h>
#include <queue>

#include "Backend/TensorMemory.h"
#include "Tensor/Tensor.h"

namespace TensorFrost {

bool IsBoundary(const Node* input, const Node* output, int arg_index,
                ArgType arg_type);

class Kernel {
 public:
	Node* root;
	map<Node*, int> variables;
	map<Node*, int> memory;
	Arguments shape;
	int dim = 0;

	int kernel_id_;
	string kernel_name_;
	string generated_code_;
};

class Program {
 public:
	IR* ir_;
	vector<Kernel> kernels_;
	function<void()> unload_callback;
	string generated_code_;
	string main_function_;
	string program_name = "TensorProgram";

	function<main_func> execute_callback;

	explicit Program(IR* ir) : ir_(ir) {}

	void AddKernel(Node* kernel_node, map<Node*, int> variables, map<Node*, int> memory,
	               Arguments shape, int dim)
	{
		kernels_.push_back(
		    {kernel_node, std::move(variables), std::move(memory), std::move(shape), dim});
	}
};

Program* GenerateProgram(IR* ir);

string GetOperationListing(const IR&, bool compact = false,
                           map<Node*, string> invalid = {});

}  // namespace TensorFrost