#pragma once

#include <functional>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>

#include "TensorMemory.h"
#include "ProgramExecutor.h"
#include "Backends\VM\Memory.h"

namespace TensorFrost {

using namespace std;

extern TensorMemoryManager* GlobalMemoryManager;

enum class BackendType {
    CPU_VM,
    WGPU,
};

void InitializeMemoryManager(/*BackendType backendType*/);

}  // namespace TensorFrost