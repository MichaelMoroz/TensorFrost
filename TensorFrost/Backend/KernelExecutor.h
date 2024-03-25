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


class KernelExecutor
{
	int global_kernel_id = 0;
 public:
	unordered_map<int, function<void()>> kernel_map;

	KernelExecutor() = default;

	void AddKernel(int kernel_id, function<void()> kernel)
	{
		kernel_map[kernel_id] = kernel;
	}

	void ExecuteKernel(int kernel_id)
	{
		kernel_map[kernel_id]();
	}

	int GenerateKernelID() { 
		return global_kernel_id++; 
	}
};

}  // namespace TensorFrost