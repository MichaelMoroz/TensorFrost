#pragma once

#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../KernelManager.h"

namespace TensorFrost {

class CpuKernelManager : public KernelManager {
	unordered_map<size_t, cpu_dispatch_func*> kernel_functions;
 public:

	void AddKernelFunction(Kernel* kernel, cpu_dispatch_func* func)	{ 
		kernel_functions[kernel->kernel_id_] = func;
	}

	cpu_dispatch_func* GetKernel(size_t id) {
		return kernel_functions[id];
	}

	virtual void DispatchKernel(TFDispatchInfo info) override
	{	
		CpuMemoryManager* memory_manager = (CpuMemoryManager*)global_memory_manager;
		cpu_dispatch_func* func = kernel_functions[info.kernel_id];
		//get memory pointers
		uint** memory = new uint*[info.tensor_count];
		for (int i = 0; i < (int)info.tensor_count; i++) {
			memory[i] = memory_manager->GetNativeBuffer(&info.tensors[i]);
		}
		func(info.variables, memory, info.work_group_count);
		delete[] memory;
	}
};

}  // namespace TensorFrost	