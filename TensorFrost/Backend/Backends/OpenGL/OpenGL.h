#pragma once

#include "glad/gl.h"
#include "GLFW/glfw3.h"
#include "KernelCompiler.h"
#include "Memory.h"
#include "KernelManager.h"

namespace TensorFrost {

void StartOpenGL();

void StopOpenGL();

}  // namespace TensorFrost