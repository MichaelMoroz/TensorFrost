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
#include "RenderDoc.h"

namespace TensorFrost {

using namespace std;

enum class BackendType {
	CPU,
	Vulkan,
	OpenGL,
	CodeGen,
	NotInitialized
};

enum class CodeGenLang {
	CPP,
	HLSL,
	GLSL,
	None,
};

extern BackendType current_backend;
extern CodeGenLang current_kernel_lang;
extern CodeGenLang current_main_lang;

vector<TFTensor*> ExecuteProgram(
    Program* program, vector<TFTensor*> inputs);

void InitializeBackend(BackendType backendType, const string& compilerPath, CodeGenLang kernelType);

void CompileKernels(Program* program);

}  // namespace TensorFrost