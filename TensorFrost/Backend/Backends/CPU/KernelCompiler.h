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

namespace TensorFrost {

using namespace std;

extern std::string C_COMPILER_PATH;

void compileLibrary();

void loadLibraryWin();

}  // namespace TensorFrost