#include "Definitions/VulkanBindings.h"
#include "VulkanInterface.h"

#include <cfloat>
#include <optional>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace TensorFrost {

void VulkanDefinitions(py::module_& m) {
    py::class_<PyBuffer>(m, "Buffer", "Vulkan-backed storage buffer exposed to Python.")
        .def(py::init<size_t, size_t, bool>(),
             py::arg("count"), py::arg("dtype_size"), py::arg("read_only") = false,
             "Create a buffer sized for `count` elements of size `dtype_size`.")
        .def_property_readonly("size", &PyBuffer::byteSize, "Total size of the buffer in bytes.")
        .def_property_readonly("count", &PyBuffer::elementCapacity,
                               "Maximum number of elements the buffer can hold for the configured dtype size.")
        .def_property_readonly("read_only", &PyBuffer::isReadOnly,
                               "Whether the buffer is flagged as read-only for compute kernels.")
        .def("setData", &PyBuffer::setData, py::arg("data"), py::arg("offset") = 0,
             "Upload data from a NumPy array or bytes-like object into the buffer.")
        .def("getData",
             [](const PyBuffer& self, const py::object& dtype, const py::object& count, size_t offset) {
                 return self.getData(dtype, count, offset);
             },
             py::arg("dtype") = py::none(), py::arg("count") = py::none(), py::arg("offset") = 0,
             "Download data from the buffer into a newly allocated NumPy array.")
        .def("release", &PyBuffer::release,
             "Explicitly destroy the underlying Vulkan buffer and release its memory.");

    m.def("createBuffer",
          [](size_t count, size_t dtypeSize, bool readOnly) {
              return PyBuffer(count, dtypeSize, readOnly);
          },
          py::arg("count"), py::arg("dtype_size"), py::arg("read_only") = false,
          py::return_value_policy::move,
          "Convenience helper to construct a :class:`Buffer` without calling the class directly.");

    py::class_<PyComputeProgram>(m, "ComputeProgram",
                                 "Compiled compute pipeline that can be dispatched on the GPU.")
        .def_property_readonly("readonly_count", &PyComputeProgram::readonlyCount,
                               "Number of read-only storage buffers expected by the program.")
        .def_property_readonly("readwrite_count", &PyComputeProgram::readwriteCount,
                               "Number of read-write storage buffers expected by the program.")
        .def_property_readonly("push_constant_size", &PyComputeProgram::pushConstantSize,
                               "Size in bytes of the push-constant block expected by this program (0 if unused).")
        .def("run", &PyComputeProgram::run,
             py::arg("readonly_buffers"),
             py::arg("readwrite_buffers"),
             py::arg("group_count"),
             py::arg("push_constants") = py::none(),
             "Dispatch the compute pipeline with the provided buffers, workgroup count, and optional push constants.")
        .def("release", &PyComputeProgram::release,
             "Explicitly destroy the underlying Vulkan pipeline and associated resources.");

    m.def("createComputeProgramFromSlang",
          [](const std::string& moduleName,
             const std::string& source,
             const std::string& entry,
             uint32_t roCount,
             uint32_t rwCount,
             uint32_t pushConstantSize) {
              return MakeComputeProgramFromSlang(
                  moduleName, source, entry, roCount, rwCount, pushConstantSize);
          },
          py::arg("module_name"), py::arg("source"), py::arg("entry"),
          py::arg("ro_count"), py::arg("rw_count"),
          py::arg("push_constant_size") = 0,
          py::return_value_policy::move,
          "Compile a Slang module to SPIR-V and wrap it in a :class:`ComputeProgram`.");

    py::class_<PyWindow>(m, "Window",
                         "GLFW-backed Vulkan swapchain window for presenting compute output.")
        .def(py::init<int, int, std::string>(), py::arg("width"), py::arg("height"), py::arg("title"),
             "Create a window with an attached Vulkan swapchain.")
        .def_property_readonly("size", &PyWindow::size,
                               "Current window extent as a tuple ``(width, height)``.")
        .def_property_readonly("format", &PyWindow::format,
                               "Pixel format of the swapchain image as a Vulkan enum value.")
        .def("isOpen", &PyWindow::isOpen,
             "Return ``True`` while the window is alive and the user has not closed it.")
        .def("drawBuffer", &PyWindow::drawBuffer,
             py::arg("buffer"), py::arg("width"), py::arg("height"), py::arg("offset") = 0,
             "Copy a buffer of packed pixels onto the swapchain.")
        .def("present", &PyWindow::present,
             "Present the current frame without uploading new pixels.")
        .def("close", &PyWindow::close,
             "Destroy the window and release its swapchain resources.")
        .def("imgui_begin", &PyWindow::imguiBegin,
             py::arg("name"), py::arg("open") = py::none(), py::arg("flags") = 0,
             "Begin a new ImGui window, returning (visible, open_flag_or_None).")
        .def("imgui_end", &PyWindow::imguiEnd,
             "End the current ImGui window.")
        .def("imgui_text", &PyWindow::imguiText,
             py::arg("text"),
             "Add text to the current ImGui window.")
        .def("imgui_button", &PyWindow::imguiButton,
             py::arg("label"),
             "Add a button and return True when pressed.")
        .def("imgui_checkbox", &PyWindow::imguiCheckbox,
             py::arg("label"), py::arg("value"),
             "Add a checkbox and return the updated value.")
        .def("imgui_slider_int", &PyWindow::imguiSliderInt,
             py::arg("label"), py::arg("value"), py::arg("min"), py::arg("max"),
             "Slider that returns the updated integer value.")
        .def("imgui_slider_float", &PyWindow::imguiSliderFloat,
             py::arg("label"), py::arg("value"), py::arg("min"), py::arg("max"),
             "Slider that returns the updated float value.")
        .def("imgui_plot_lines", &PyWindow::imguiPlotLines,
             py::arg("label"), py::arg("values"), py::arg("values_offset") = 0,
             py::arg("overlay_text") = "", py::arg("scale_min") = FLT_MAX,
             py::arg("scale_max") = FLT_MAX,
             py::arg("graph_size") = py::make_tuple(0.0f, 0.0f),
             py::arg("stride") = sizeof(float),
             "Plot a sequence of values as lines.")
        .def("imgui_scale_all_sizes", &PyWindow::imguiScaleAllSizes,
             py::arg("scale"),
             "Scale all ImGui sizes by a factor.")
        .def("imgui_add_background_text", &PyWindow::imguiAddBackgroundText,
             py::arg("text"), py::arg("pos"), py::arg("color"),
             "Draw text in the background draw list.")
        .def("imgui_same_line", &PyWindow::imguiSameLine,
             py::arg("offset_from_start_x") = 0.0f, py::arg("spacing") = -1.0f,
             "Place the next item on the same horizontal line.")
        .def("imgui_separator", &PyWindow::imguiSeparator,
             "Insert a separator line between items.")
        .def("imgui_spacing", &PyWindow::imguiSpacing,
             "Insert vertical spacing between items.")
        .def("imgui_indent", &PyWindow::imguiIndent,
             py::arg("indent_w") = 0.0f,
             "Increase the current horizontal indent.")
        .def("imgui_unindent", &PyWindow::imguiUnindent,
             py::arg("indent_w") = 0.0f,
             "Decrease the current horizontal indent.")
        .def("imgui_begin_child", &PyWindow::imguiBeginChild,
             py::arg("id"), py::arg("size") = py::none(), py::arg("border") = false, py::arg("flags") = 0,
             "Begin a child region and return True if visible. Always pair with :meth:`imgui_end_child` even when the return value is False.")
        .def("imgui_end_child", &PyWindow::imguiEndChild,
             "End the current child region.")
        .def("imgui_text_wrapped", &PyWindow::imguiTextWrapped,
             py::arg("text"),
             "Render wrapped text within the current column width.")
        .def("imgui_text_colored", &PyWindow::imguiTextColored,
             py::arg("color"), py::arg("text"),
             "Render text with the given RGBA color.")
        .def("imgui_bullet_text", &PyWindow::imguiBulletText,
             py::arg("text"),
             "Render text preceded by a bullet.")
        .def("imgui_input_text", &PyWindow::imguiInputText,
             py::arg("label"), py::arg("value"), py::arg("buffer_length") = 0, py::arg("flags") = 0,
             "Input text returning (modified, value).")
        .def("imgui_input_int", &PyWindow::imguiInputInt,
             py::arg("label"), py::arg("value"), py::arg("step") = 1, py::arg("step_fast") = 100, py::arg("flags") = 0,
             "Integer input returning the updated value.")
        .def("imgui_input_float", &PyWindow::imguiInputFloat,
             py::arg("label"), py::arg("value"), py::arg("step") = 0.0f, py::arg("step_fast") = 0.0f,
             py::arg("format") = "%.3f", py::arg("flags") = 0,
             "Float input returning the updated value.")
        .def("imgui_color_edit3", &PyWindow::imguiColorEdit3,
             py::arg("label"), py::arg("color"), py::arg("flags") = 0,
             "Color editor returning (modified, rgb tuple).")
        .def("imgui_color_edit4", &PyWindow::imguiColorEdit4,
             py::arg("label"), py::arg("color"), py::arg("flags") = 0,
             "Color editor returning (modified, rgba tuple).")
        .def("imgui_begin_main_menu_bar", &PyWindow::imguiBeginMainMenuBar,
             "Begin the global main menu bar, returning True if it is visible.")
        .def("imgui_end_main_menu_bar", &PyWindow::imguiEndMainMenuBar,
             "End the global main menu bar.")
        .def("imgui_begin_menu_bar", &PyWindow::imguiBeginMenuBar,
             "Begin a menu bar on the current window, returning True if visible.")
        .def("imgui_end_menu_bar", &PyWindow::imguiEndMenuBar,
             "End the current menu bar.")
        .def("imgui_begin_menu", &PyWindow::imguiBeginMenu,
             py::arg("label"), py::arg("enabled") = true,
             "Begin a menu entry and return True if it is open.")
        .def("imgui_end_menu", &PyWindow::imguiEndMenu,
             "End the current menu entry.")
        .def("imgui_menu_item", &PyWindow::imguiMenuItem,
             py::arg("label"), py::arg("shortcut") = py::none(), py::arg("selected") = false, py::arg("enabled") = true,
             "Create a menu item and return True when activated.")
        .def("imgui_open_popup", &PyWindow::imguiOpenPopup,
             py::arg("id"), py::arg("popup_flags") = 0,
             "Open a popup window by identifier.")
        .def("imgui_begin_popup", &PyWindow::imguiBeginPopup,
             py::arg("id"), py::arg("flags") = 0,
             "Begin a popup window, returning True if it is open.")
        .def("imgui_begin_popup_modal", &PyWindow::imguiBeginPopupModal,
             py::arg("name"), py::arg("open") = py::none(), py::arg("flags") = 0,
             "Begin a modal popup, returning (visible, open_flag_or_None).")
        .def("imgui_end_popup", &PyWindow::imguiEndPopup,
             "End the current popup window.")
        .def("imgui_close_current_popup", &PyWindow::imguiCloseCurrentPopup,
             "Close the current popup window.")
        .def("imgui_push_style_color", &PyWindow::imguiPushStyleColor,
             py::arg("index"), py::arg("color"),
             "Push a style color onto the stack.")
        .def("imgui_pop_style_color", &PyWindow::imguiPopStyleColor,
             py::arg("count") = 1,
             "Pop style colors from the stack.")
        .def("imgui_push_style_var_float", &PyWindow::imguiPushStyleVarFloat,
             py::arg("index"), py::arg("value"),
             "Push a float style variable onto the stack.")
        .def("imgui_push_style_var_vec2", &PyWindow::imguiPushStyleVarVec2,
             py::arg("index"), py::arg("value"),
             "Push a 2D vector style variable onto the stack.")
        .def("imgui_pop_style_var", &PyWindow::imguiPopStyleVar,
             py::arg("count") = 1,
             "Pop style variables from the stack.")
        .def("imgui_get_font_global_scale", &PyWindow::imguiGetFontGlobalScale,
             "Get the global font scale factor.")
        .def("imgui_set_font_global_scale", &PyWindow::imguiSetFontGlobalScale,
             py::arg("scale"),
             "Set the global font scale factor.")
        .def("imgui_get_style_color_vec4", &PyWindow::imguiGetStyleColorVec4,
             py::arg("index"),
             "Get a style color as an RGBA tuple.")
        .def("imgui_set_style_color_vec4", &PyWindow::imguiSetStyleColorVec4,
             py::arg("index"), py::arg("color"),
             "Set a style color from an RGBA tuple.")
        .def("mouse_position", &PyWindow::mousePosition,
             "Get the current mouse position in window coordinates.")
        .def("is_mouse_button_pressed", &PyWindow::isMouseButtonPressed,
             py::arg("button"),
             "Return True if the specified mouse button is pressed.")
        .def("imgui_want_capture_mouse", &PyWindow::imguiWantCaptureMouse,
             "Return True if ImGui wants to capture mouse input this frame.")
        .def("consume_scroll_delta", &PyWindow::consumeScrollDelta,
             "Consume the accumulated scroll delta as a tuple ``(x, y)`` and reset it to zero.");

    m.def("createWindow",
          [](int width, int height, const std::string& title) {
              return PyWindow(width, height, title);
          },
          py::arg("width"), py::arg("height"), py::arg("title"),
          py::return_value_policy::move,
          "Convenience helper to construct a :class:`Window` without calling the class directly.");
}

}  // namespace TensorFrost
