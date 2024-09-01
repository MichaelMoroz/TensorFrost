#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>
#include <Frontend/Python/PyTensorMemory.h>

#include "Backend/RenderDoc.h"

namespace TensorFrost {

void WindowDefinitions(py::module& m) {
	py::module window = m.def_submodule("window", "Window functions");

	window.def(
	    "show",
	    [](int width, int height, string title) {
		    ShowWindow(width, height, title.c_str());
	    },
	    "Show the memory manager window");

	window.def(
	    "hide", []() { HideWindow(); }, "Hide the memory manager window");

	window.def(
	    "render_frame", [](const PyTensorMemory& t) { RenderFrame(t.tensor_); },
	    "Render a frame from the tensor memory");

	window.def("render_frame", []() { RenderFrame(nullptr); },
	    "Render an empty frame");

	window.def(
	    "should_close", []() { return WindowShouldClose(); },
	    "Check if the window should close");

	window.def(
	    "get_mouse_position", []() { return GetMousePosition(); },
	    "Get the current mouse position");

	window.def(
	    "get_size", []() { return GetWindowSize(); },
	    "Get the current window size");

	window.def(
	    "is_mouse_button_pressed",
	    [](int button) { return IsMouseButtonPressed(button); },
	    "Check if a mouse button is pressed");

	window.def(
	    "is_key_pressed", [](int key) { return IsKeyPressed(key); },
	    "Check if a key is pressed");

	window.attr("MOUSE_BUTTON_0") = GLFW_MOUSE_BUTTON_1;
	window.attr("MOUSE_BUTTON_1") = GLFW_MOUSE_BUTTON_2;
	window.attr("MOUSE_BUTTON_2") = GLFW_MOUSE_BUTTON_3;

	window.attr("KEY_SPACE") = GLFW_KEY_SPACE;
	window.attr("KEY_APOSTROPHE") = GLFW_KEY_APOSTROPHE;
	window.attr("KEY_COMMA") = GLFW_KEY_COMMA;
	window.attr("KEY_MINUS") = GLFW_KEY_MINUS;
	window.attr("KEY_PERIOD") = GLFW_KEY_PERIOD;

	window.attr("KEY_A") = GLFW_KEY_A;
	window.attr("KEY_B") = GLFW_KEY_B;
	window.attr("KEY_C") = GLFW_KEY_C;
	window.attr("KEY_D") = GLFW_KEY_D;
	window.attr("KEY_E") = GLFW_KEY_E;
	window.attr("KEY_F") = GLFW_KEY_F;
	window.attr("KEY_G") = GLFW_KEY_G;
	window.attr("KEY_H") = GLFW_KEY_H;
	window.attr("KEY_I") = GLFW_KEY_I;
	window.attr("KEY_J") = GLFW_KEY_J;
	window.attr("KEY_K") = GLFW_KEY_K;
	window.attr("KEY_L") = GLFW_KEY_L;
	window.attr("KEY_M") = GLFW_KEY_M;
	window.attr("KEY_N") = GLFW_KEY_N;
	window.attr("KEY_O") = GLFW_KEY_O;
	window.attr("KEY_P") = GLFW_KEY_P;
	window.attr("KEY_Q") = GLFW_KEY_Q;
	window.attr("KEY_R") = GLFW_KEY_R;
	window.attr("KEY_S") = GLFW_KEY_S;
	window.attr("KEY_T") = GLFW_KEY_T;
	window.attr("KEY_U") = GLFW_KEY_U;
	window.attr("KEY_V") = GLFW_KEY_V;
	window.attr("KEY_W") = GLFW_KEY_W;
	window.attr("KEY_X") = GLFW_KEY_X;
	window.attr("KEY_Y") = GLFW_KEY_Y;
	window.attr("KEY_Z") = GLFW_KEY_Z;

	py::module imgui = m.def_submodule("imgui", "ImGui functions");

	imgui.def(
	    "begin", [](string name) { ImGuiBegin(name); },
	    "Begin a new ImGui window");

	imgui.def("end", []() { ImGuiEnd(); }, "End the current ImGui window");

	imgui.def(
	    "text", [](string text) { ImGuiText(text); },
	    "Add text to the current ImGui window");

	imgui.def("slider",
		[](string text, int* value, int min, int max) {
		      ImGuiSlider(text, value, min, max);
			  return *value;
	      }, "Add a slider to the current ImGui window");

	imgui.def("slider", [](string text, float* value, float min, float max) {
		      ImGuiSlider(text, value, min, max);
			  return *value;
	      }, "Add a slider to the current ImGui window");

	imgui.def("button", [](string text) { return ImGuiButton(text); },
			      "Add a button to the current ImGui window");

	imgui.def("checkbox", [](string text, bool* value) {
		       ImGuiCheckbox(text, value);
			   return *value;
	      }, "Add a checkbox to the current ImGui window");

	imgui.def("plotlines", [](string label, py::array_t<float> values, int values_offset, string overlay_text, float scale_min, float scale_max, py::tuple graph_size, int stride) {
		ImGuiPlotLines(label.c_str(), values.data(), (int)values.size(), values_offset, overlay_text.c_str(), scale_min, scale_max, ImVec2(graph_size[0].cast<float>(), graph_size[1].cast<float>()), stride);
	}, py::arg("label"), py::arg("values"), py::arg("values_offset") = 0, py::arg("overlay_text") = "", py::arg("scale_min") = FLT_MAX, py::arg("scale_max") = FLT_MAX, py::arg("graph_size") = py::make_tuple(0.0f, 0.0f), py::arg("stride") = sizeof(float));

	imgui.def("scale_all_sizes", [](float scale) { ImGuiScaleAllSizes(scale); },
	      "Scale all ImGui sizes by a factor");

	imgui.def("add_background_text", [](string text, py::tuple pos, py::tuple color) {
		ImGuiAddBackgroundText(text, ImVec2(pos[0].cast<float>(), pos[1].cast<float>()), ImVec4(color[0].cast<float>(), color[1].cast<float>(), color[2].cast<float>(), color[3].cast<float>()));
	}, py::arg("text"), py::arg("pos"), py::arg("color"));

	//renderdoc
	m.def("renderdoc_start_capture", []() { StartRenderDocCapture(); },
	      "Start a RenderDoc capture");
	m.def("renderdoc_end_capture", []() { EndRenderDocCapture(); },
	      "End a RenderDoc capture");
}

}  // namespace TensorFrost