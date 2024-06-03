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
	map<Node*, size_t> variables;
	map<Node*, size_t> read_write_memory;
	map<Node*, size_t> read_only_memory;
	NodeArguments shape;

	uint kernel_id_;
	string kernel_name_;
	string generated_code_;

	map<Node*, size_t> GetMemoryBindings() {
		map<Node*, size_t> result;
		for (auto& mem : read_write_memory) {
			result[mem.first] = mem.second;
		}
		for (auto& mem : read_only_memory) {
			result[mem.first] = mem.second + read_write_memory.size();
		}
		return result;
	}
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

	void AddKernel(Node* kernel_node, map<Node*, size_t> variables, map<Node*, size_t> read_write, map<Node*, size_t> read_only,
	               NodeArguments shape)
	{
		kernels_.push_back(
		    {kernel_node, std::move(variables), std::move(read_write), std::move(read_only), std::move(shape)});
	}
};

Program* GenerateProgram(IR* ir);

string GetOperationListing(const IR&, bool compact = false,
                           map<Node*, string> invalid = {});

}  // namespace TensorFrost