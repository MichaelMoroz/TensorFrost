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
	unordered_map<int, cpu_dispatch_func*> kernel_functions;
 public:

	void AddKernelFunction(Kernel* kernel, cpu_dispatch_func* func)	{ 
		kernel_functions[kernel->kernel_id_] = func;
	}

	cpu_dispatch_func* GetKernel(int id) { 
		return kernel_functions[id];
	}

	virtual void DispatchKernel(DispatchInfo info) override
	{	
		CpuMemoryManager* memory_manager = (CpuMemoryManager*)global_memory_manager;
		cpu_dispatch_func* func = kernel_functions[info.kernel_id];
		//get memory pointer
		uint* memory = memory_manager->memory.data();
		//get memory offsets
		uint* offsets = new uint[info.tensor_count];
		for (int i = 0; i < (int)info.tensor_count; i++) {
			offsets[i] = info.tensors[i].offset;
		}
		uint* variables = info.variables;
		uint* shape = info.dispatch_shape;
		func(variables, offsets, memory, shape);
		delete[] offsets;
	}
};

}  // namespace TensorFrost	