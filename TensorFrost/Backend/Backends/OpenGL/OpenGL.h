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

void RenderFrame(const TFTensor& tensor);

bool WindowShouldClose();

pair<double, double> GetMousePosition();
pair<int, int> GetWindowSize();

bool IsMouseButtonPressed(int button);
bool IsKeyPressed(int key);

void ImGuiBegin(std::string name);
void ImGuiEnd();

void ImGuiText(const std::string& text);
void ImGuiSlider(std::string text, int* value, int min, int max);
void ImGuiSlider(std::string text, float* value, float min, float max);
bool ImGuiCheckbox(std::string text, bool* value);
bool ImGuiButton(std::string text);

void ImGuiPlotLines(const char* label, const float* values, int values_count, int values_offset = 0, const char* overlay_text = NULL, float scale_min = FLT_MAX, float scale_max = FLT_MAX, ImVec2 graph_size = ImVec2(0, 0), int stride = sizeof(float));

void ImGuiScaleAllSizes(float scale);

void StartDebugRegion(const std::string& name);
void EndDebugRegion();

}  // namespace TensorFrost