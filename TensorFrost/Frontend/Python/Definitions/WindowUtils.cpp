#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

void WindowDefinitions(py::module& m) {
	m.def(
	    "show_window",
	    [](int width, int height, string title) {
		    ShowWindow(width, height, title.c_str());
	    },
	    "Show the memory manager window");

	m.def(
	    "hide_window", []() { HideWindow(); }, "Hide the memory manager window");

	m.def(
	    "render_frame", [](const TensorMemory& t) { RenderFrame(t); },
	    "Render a frame from the tensor memory");

	m.def(
	    "window_should_close", []() { return WindowShouldClose(); },
	    "Check if the window should close");

	m.def(
	    "get_mouse_position", []() { return GetMousePosition(); },
	    "Get the current mouse position");

	m.def(
	    "get_window_size", []() { return GetWindowSize(); },
	    "Get the current window size");

	m.def(
	    "is_mouse_button_pressed",
	    [](int button) { return IsMouseButtonPressed(button); },
	    "Check if a mouse button is pressed");

	m.def(
	    "is_key_pressed", [](int key) { return IsKeyPressed(key); },
	    "Check if a key is pressed");

	m.def(
	    "imgui_begin", [](string name) { ImGuiBegin(name); },
	    "Begin a new ImGui window");

	m.def("imgui_end", []() { ImGuiEnd(); }, "End the current ImGui window");

	m.def(
	    "imgui_text", [](string text) { ImGuiText(text); },
	    "Add text to the current ImGui window");

	m.def("imgui_slider",
		[](string text, int* value, int min, int max) {
		      ImGuiSlider(text, value, min, max);
			  return *value;
	      },
			      "Add a slider to the current ImGui window");

	m.def("imgui_slider", [](string text, float* value, float min, float max) {
		      ImGuiSlider(text, value, min, max);
			  return *value;
	      },
					      "Add a slider to the current ImGui window");

	m.def("imgui_button", [](string text) { return ImGuiButton(text); },
			      "Add a button to the current ImGui window");

	m.attr("MOUSE_BUTTON_0") = GLFW_MOUSE_BUTTON_1;
	m.attr("MOUSE_BUTTON_1") = GLFW_MOUSE_BUTTON_2;
	m.attr("MOUSE_BUTTON_2") = GLFW_MOUSE_BUTTON_3;

	m.attr("KEY_SPACE") = GLFW_KEY_SPACE;
	m.attr("KEY_APOSTROPHE") = GLFW_KEY_APOSTROPHE;
	m.attr("KEY_COMMA") = GLFW_KEY_COMMA;
	m.attr("KEY_MINUS") = GLFW_KEY_MINUS;
	m.attr("KEY_PERIOD") = GLFW_KEY_PERIOD;
	
	m.attr("KEY_A") = GLFW_KEY_A;
	m.attr("KEY_B") = GLFW_KEY_B;
	m.attr("KEY_C") = GLFW_KEY_C;
	m.attr("KEY_D") = GLFW_KEY_D;
	m.attr("KEY_E") = GLFW_KEY_E;
	m.attr("KEY_F") = GLFW_KEY_F;
	m.attr("KEY_G") = GLFW_KEY_G;
	m.attr("KEY_H") = GLFW_KEY_H;
	m.attr("KEY_I") = GLFW_KEY_I;
	m.attr("KEY_J") = GLFW_KEY_J;
	m.attr("KEY_K") = GLFW_KEY_K;
	m.attr("KEY_L") = GLFW_KEY_L;
	m.attr("KEY_M") = GLFW_KEY_M;
	m.attr("KEY_N") = GLFW_KEY_N;
	m.attr("KEY_O") = GLFW_KEY_O;
	m.attr("KEY_P") = GLFW_KEY_P;
	m.attr("KEY_Q") = GLFW_KEY_Q;
	m.attr("KEY_R") = GLFW_KEY_R;
	m.attr("KEY_S") = GLFW_KEY_S;
	m.attr("KEY_T") = GLFW_KEY_T;
	m.attr("KEY_U") = GLFW_KEY_U;
	m.attr("KEY_V") = GLFW_KEY_V;
	m.attr("KEY_W") = GLFW_KEY_W;
	m.attr("KEY_X") = GLFW_KEY_X;
	m.attr("KEY_Y") = GLFW_KEY_Y;
	m.attr("KEY_Z") = GLFW_KEY_Z;
}

}  // namespace TensorFrost