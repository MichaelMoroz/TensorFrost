#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

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

pair<int, int> GetMousePosition();

bool IsMouseButtonPressed(int button);
bool IsKeyPressed(int key);

}  // namespace TensorFrost