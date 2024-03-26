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
	unordered_map<int, Kernel*> kernel_map;
	int global_kernel_id = 0;
 public:
	KernelManager() = default;

	virtual void DispatchKernel(DispatchInfo info) = 0;

	void AddKernelID(Kernel* kernel) { 
		kernel->kernel_id_ = global_kernel_id++; 
		kernel_map[kernel->kernel_id_] = kernel;
	}
};

extern KernelManager* global_kernel_manager;

}  // namespace TensorFrost