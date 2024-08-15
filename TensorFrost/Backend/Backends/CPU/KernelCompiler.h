#pragma once

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#ifndef NOMINMAX
#define NOMINMAX
#endif

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#include <sys/wait.h>
#endif

#include "Backend/Backends/CPU/Memory.h"
#include "Backend/CodeGen/Generators.h"
#include "Backend/KernelManager.h"
#include "Backend/TensorMemory.h"
#include "Compiler/KernelGen.h"

namespace TensorFrost {

using namespace std;

extern std::string kernel_compile_options;

void CompileAndLoadKernelModule(Program* program, size_t program_id);

}  // namespace TensorFrost