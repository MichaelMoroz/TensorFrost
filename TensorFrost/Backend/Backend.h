#pragma once

#include <functional>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>

#include "TensorMemory.h"
#include "KernelExecutor.h"

#include "Backends\CPU\CPU.h"

namespace TensorFrost {

using namespace std;

extern TensorMemoryManager* GlobalMemoryManager;

enum class BackendType {
    CPU,
    WGPU,
};

vector<TensorMemory*> ExecuteProgram(Program* program, vector<TensorMemory*> inputs);

void InitializeBackend(BackendType backendType, string compilerPath = "");

}  // namespace TensorFrost