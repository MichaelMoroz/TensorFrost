#pragma once

#include <functional>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>

#define NOMINMAX
#include <windows.h>

#include "IR/KernelGen.h"
#include "Backend/KernelExecutor.h"
#include "Backend/CodeGen/Generators.h"
#include "Backend/TensorMemory.h"
#include "Backend/Backends/CPU/Memory.h"

namespace TensorFrost {

using namespace std;

extern std::string C_COMPILER_PATH;

void CompileAndLoadKernel(Program* program);

}  // namespace TensorFrost