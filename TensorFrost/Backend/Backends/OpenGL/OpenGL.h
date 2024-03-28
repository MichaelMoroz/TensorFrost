#pragma once

#include "glad/gl.h"
#include "GLFW/glfw3.h"

#include "Memory.h"
#include "KernelManager.h"

namespace TensorFrost {

void StartOpenGL();

void StopOpenGL();

void ShowWindow(int width, int height, const char* title);
void HideWindow();

void RenderFrame(const TensorMemory& tensor);

bool WindowShouldClose();

}  // namespace TensorFrost