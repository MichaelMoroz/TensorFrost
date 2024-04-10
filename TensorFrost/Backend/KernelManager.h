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
	unordered_map<int, Kernel*> kernel_map;
	int global_kernel_id = 0;
 public:
	KernelManager() = default;

	virtual void DispatchKernel(DispatchInfo info) = 0;

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

	vector<string> GetAllKernels() {
		vector<string> kernels;
		kernels.resize(kernel_map.size());
		for (auto& kernel : kernel_map) {
			kernels[kernel.first] = kernel.second->generated_code_;
		}
		return kernels;
	}
};

extern KernelManager* global_kernel_manager;

}  // namespace TensorFrost