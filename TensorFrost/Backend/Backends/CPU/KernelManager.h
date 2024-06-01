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
		uint32_t** memory = new uint32_t*[info.read_write_count];
		for (size_t i = 0; i < info.read_write_count; i++) {
			memory[i] = memory_manager->GetNativeBuffer(&info.read_write_tensors[i]);
		}
		func(info.variables, memory, (uint)info.work_group_count);
		delete[] memory;
	}
};

}  // namespace TensorFrost	