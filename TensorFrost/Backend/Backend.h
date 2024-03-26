#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Backends/CPU/CPU.h"
#include "Backends/OpenGL/OpenGL.h"
#include "CodeGen/Generators.h"
#include "KernelManager.h"
#include "TensorMemory.h"

namespace TensorFrost {

using namespace std;

enum class BackendType {
	CPU,
	Vulkan,
	OpenGL,
	NotInitialized
};

extern BackendType current_backend;

vector<TensorMemory*> ExecuteProgram(
    Program* program, vector<TensorMemory*> inputs);

void InitializeBackend(BackendType backendType,
                       const string& compilerPath = "");

void CompileKernels(Program* program);

}  // namespace TensorFrost