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

void Finish();

void RenderFrame(const TensorMemory& tensor);

bool WindowShouldClose();

pair<double, double> GetMousePosition();
pair<int, int> GetWindowSize();

bool IsMouseButtonPressed(int button);
bool IsKeyPressed(int key);

void ImGuiBegin(std::string name);
void ImGuiEnd();

void ImGuiText(std::string text);
void ImGuiSlider(std::string text, int* value, int min, int max);
void ImGuiSlider(std::string text, float* value, float min, float max);

bool ImGuiButton(std::string text);

void StartDebugRegion(const std::string& name);
void EndDebugRegion();

}  // namespace TensorFrost