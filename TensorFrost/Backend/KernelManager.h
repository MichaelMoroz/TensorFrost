#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "IR/KernelGen.h"
#include "TensorMemory.h"

namespace TensorFrost {

class KernelManager
{
	unordered_set<Program*> programs;
	unordered_map<uint, Kernel*> kernel_map;
	int global_kernel_id = 0;
 public:
	KernelManager() = default;

	virtual void DispatchKernel(TFDispatchInfo info) = 0;

	void AddKernelID(Program* program, Kernel* kernel) { 
		programs.insert(program);
		kernel->kernel_id_ = global_kernel_id++; 
		kernel_map[kernel->kernel_id_] = kernel;
	}

	vector<string> GetAllMainFunctions() {
		vector<string> main_functions;
		for (auto& program : programs) {
			main_functions.push_back(program->main_function_);
		}
		return main_functions;
	}

	vector<tuple<string, vector<tuple<string, int, string>>>> GetAllKernels() {
		vector<tuple<string, vector<tuple<string, int, string>>>> kernels;
		kernels.resize(kernel_map.size());
		for (auto& kernel : kernel_map) {
			vector<tuple<string, int, string>> args;
			map<Node*, size_t> memory_bindings = kernel.second->GetMemoryBindings();
			for (auto& [mem_node, binding] : memory_bindings) {
				string name = mem_node->var_name + "_mem";
				string type_name = "uint";
				args.push_back({name, (int)binding, type_name});
			}
			string code = kernel.second->generated_code_;
			kernels[kernel.first] = {code, args};
		}
		return kernels;
	}
};

extern KernelManager* global_kernel_manager;

}  // namespace TensorFrost