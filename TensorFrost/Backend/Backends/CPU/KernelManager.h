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
	unordered_map<int, Kernel*> kernel_map;

 public:

	void AddKernel(Kernel* kernel, cpu_dispatch_func* func)	{ 
		kernel_functions[kernel->kernel_id_] = func;
		kernel_map[kernel->kernel_id_] = kernel;
	}

	cpu_dispatch_func* GetKernel(int id) { 
		return kernel_functions[id];
	}
};

}  // namespace TensorFrost	