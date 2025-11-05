#include "Definitions/VulkanBindings.h"
#include "VulkanInterface.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

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
        .def("run", &PyComputeProgram::run,
             py::arg("readonly_buffers"), py::arg("readwrite_buffers"), py::arg("num_invocations"),
             "Dispatch the compute pipeline with the provided buffers and invocation count.")
        .def("release", &PyComputeProgram::release,
             "Explicitly destroy the underlying Vulkan pipeline and associated resources.");

    m.def("createComputeProgramFromGLSL",
          [](const std::string& source, uint32_t roCount, uint32_t rwCount) {
              return MakeComputeProgramFromGLSL(source, roCount, rwCount);
          },
          py::arg("glsl_source"), py::arg("ro_count"), py::arg("rw_count"),
          py::return_value_policy::move,
          "Compile a compute shader written in GLSL to SPIR-V and wrap it in a :class:`ComputeProgram`.");

    m.def("createComputeProgramFromSlang",
          [](const std::string& moduleName, const std::string& source, const std::string& entry, uint32_t roCount, uint32_t rwCount) {
              return MakeComputeProgramFromSlang(moduleName, source, entry, roCount, rwCount);
          },
          py::arg("module_name"), py::arg("source"), py::arg("entry"),
          py::arg("ro_count"), py::arg("rw_count"),
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
        .def("close", &PyWindow::close,
             "Destroy the window and release its swapchain resources.");

    m.def("createWindow",
          [](int width, int height, const std::string& title) {
              return PyWindow(width, height, title);
          },
          py::arg("width"), py::arg("height"), py::arg("title"),
          py::return_value_policy::move,
          "Convenience helper to construct a :class:`Window` without calling the class directly.");
}

}  // namespace TensorFrost
