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
	int global_kernel_id = 0;
 public:

	KernelManager() = default;

	virtual void DispatchKernel(DispatchInfo info) = 0;

	int GenerateKernelID() { 
		return global_kernel_id++; 
	}
};

extern KernelManager* global_kernel_manager;

}  // namespace TensorFrost