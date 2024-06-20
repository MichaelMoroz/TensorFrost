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
	unordered_map<size_t, Kernel*> kernel_map;
	size_t global_kernel_id = 0;
 public:

	KernelManager() = default;
	virtual void DispatchKernel(TFDispatchInfo info) = 0;
	void AddKernelID(Program* program, Kernel* kernel);
	vector<string> GetAllMainFunctions();

	vector<tuple<tuple<string, string, string>, vector<tuple<string, string>>>> GetAllKernels();
	Kernel* GetKernel(size_t kernel_id) { return kernel_map[kernel_id]; }
};

extern KernelManager* global_kernel_manager;

}  // namespace TensorFrost