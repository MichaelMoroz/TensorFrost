#pragma once

#include <utility>
#include <limits.h>
#include <float.h>
#include <queue>

#include "Backend/TensorMemory.h"
#include "Tensor/Tensor.h"
#include "Compiler/Implementations.h"

namespace TensorFrost {

uint GetInitialMax(TFType type);
uint GetInitialMin(TFType type);

bool IsBoundary(const Node* input, const Node* output, int arg_index,
                ArgType arg_type);

class Kernel {
 public:
	Node* root;
	map<Node*, size_t> variables;
	map<Node*, size_t> read_write_memory;
	map<Node*, size_t> read_only_memory;
	NodeArguments shape;

	size_t kernel_id_;
	string kernel_name_;
	string full_generated_code_;
	string generated_header_;
	string generated_bindings_;
	string generated_main_;

	vector<string> var_names;
	vector<string> var_types;

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

bool isConstantAndEqualTo(const Tensor* tensor, float value);
bool isConstant(const Tensor* tensor);
Tensor* ApplyMultiOP(const Tensor* a, const Tensor* b, std::function<float(float, float)> opF32, std::function<int(int, int)> opI32, std::function<uint(uint, uint)> opU32);
Tensor* ApplyUnaryOP(const Tensor* a, std::function<float(float)> opF32, std::function<int(int)> opI32, std::function<uint(uint)> opU32);

}  // namespace TensorFrost