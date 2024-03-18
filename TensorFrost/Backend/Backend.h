#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Backends/CPU/CPU.h"
#include "CodeGen/Generators.h"
#include "KernelExecutor.h"
#include "TensorMemory.h"

namespace TensorFrost {

using namespace std;

extern TensorMemoryManager* global_memory_manager;

enum class BackendType {
	CPU,
	Vulkan,
};

vector<TensorMemory*> ExecuteProgram(
    Program* program, vector<TensorMemory*> inputs);

void InitializeBackend(BackendType backendType,
                       const string& compilerPath = "");

}  // namespace TensorFrost