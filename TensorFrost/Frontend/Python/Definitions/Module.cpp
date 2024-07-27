#include <utility>
#include <vector>

#include <Frontend/Python/PyModule.h>

namespace TensorFrost {

class PyModule : public Module {
public:
    using Module::Module; // Inherit constructors

    void assert_parameters() override {
        PYBIND11_OVERRIDE(void, Module, assert_parameters);
    }

    py::object loss(py::object X, py::object Y) override {
        PYBIND11_OVERRIDE_PURE(py::object, Module, loss, X, Y);
    }

    py::object forward(py::object X) override {
        PYBIND11_OVERRIDE_PURE(py::object, Module, forward, X);
    }
};

void ModuleDefinitions(py::module& m) {
    py::class_<Parameter>(m, "Parameter")
        .def(py::init<const std::vector<int>&, TFType, bool>(), py::arg("shape"), py::arg("dtype"), py::arg("random_init") = true)
        .def_readwrite("shape", &Parameter::shape)
        .def_readwrite("dtype", &Parameter::dtype)
        .def_readwrite("random_init", &Parameter::random_init)
        .def("__repr__", [](const Parameter& p) {
            return "Parameter(shape=" + std::to_string(p.shape.size()) + ", dtype=" + std::to_string(p.dtype) + ", random_init=" + std::to_string(p.random_init) + ")";
        });

    py::class_<ParameterArray>(m, "ParameterArray")
        .def(py::init<std::string>(), py::arg("prefix"))
        .def("__getitem__", &ParameterArray::__getitem__)
        .def("__setitem__", &ParameterArray::__setitem__);

    py::class_<Module, PyModule>(m, "Module")
        .def(py::init<>())
        .def("register_parameter", &Module::register_parameter)
        .def("register_module", &Module::register_module)
        .def("register_parameter_array", &Module::register_parameter_array)
        .def("__getattr__", &Module::__getattr__)
        .def("__setattr__", &Module::__setattr__)
        .def("initialize_input", &Module::initialize_input)
        .def("initialize_parameters", &Module::initialize_parameters)
        .def("get_all_parameters", &Module::get_all_parameters)
        .def("create_input", &Module::create_input)
        .def("update_parameters", &Module::update_parameters)
        .def("assert_parameters", &Module::assert_parameters)
        .def("loss", &Module::loss)
        .def("forward", &Module::forward)
        .def_readwrite("_parameters", &Module::_parameters)
        .def_readwrite("_modules", &Module::_modules)
        .def_readwrite("_parameter_arrays", &Module::_parameter_arrays);
}

}  // namespace TensorFrost