// #include <utility>
// #include <vector>
//
// #include <Frontend/Python/PyModule.h>
// #include <Frontend/Python/PyTensorMemory.h>
//
// namespace TensorFrost {
//
// class PyModule : public Module {
// public:
//     using Module::Module; // Inherit constructors
//
//     void assert_parameters() override {
//         PYBIND11_OVERRIDE(void, Module, assert_parameters);
//     }
//
//     py::object loss(py::object X, py::object Y) override {
//         PYBIND11_OVERRIDE_PURE(py::object, Module, loss, X, Y);
//     }
//
//     py::object forward(py::object X) override {
//         PYBIND11_OVERRIDE_PURE(py::object, Module, forward, X);
//     }
// };
//
// void ModuleDefinitions(py::module& m) {
//     py::class_<Parameter>(m, "Parameter")
//         .def(py::init<const std::vector<int>&, TFDataFormat, float, float, bool>(), py::arg("shape"), py::arg("dtype") = TFType::Float, py::arg("random_scale") = -1.0f, py::arg("random_offset") = 0.0f, py::arg("optimize") = true)
//         .def_readwrite("shape", &Parameter::shape)
//         .def_readwrite("dtype", &Parameter::dtype)
//         .def_readwrite("random_scale", &Parameter::random_scale)
//         .def_readwrite("random_offset", &Parameter::random_offset)
//         .def("__repr__", [](const Parameter& p) {
//             return "Parameter(shape=" + std::to_string(p.shape.size()) + ", dtype=" + std::to_string(p.dtype.type) + "( " + std::to_string(p.dtype.size) + ") , random_scale=" + std::to_string(p.random_scale) + ", random_offset=" + std::to_string(p.random_offset) + ", optimize=" + std::to_string(p.optimize) + ")";
//         });
//
//     py::class_<ParameterArray>(m, "ParameterArray")
//         .def(py::init<>())
//         .def("__getitem__", &ParameterArray::getitem)
//         .def("__setitem__", &ParameterArray::setitem)
//         .def("items", &ParameterArray::items);
//
//     py::class_<Module, PyModule>(m, "Module")
//         .def(py::init<bool>(), py::arg("requires_grad") = true)
//         .def("__getattr__", &Module::getattr)
//         .def("__setattr__", &Module::setattr)
//         .def("hasattr", &Module::hasattr)
//         .def("param_requires_grad", &Module::param_requires_grad)
//         .def("initialize_input", &Module::initialize_input)
//         .def("initialize_parameters", &Module::initialize_parameters)
//         .def("initialize_parameters_native", &Module::initialize_parameters_native)
//         .def("parameters", &Module::parameters)
//         .def("named_parameters", &Module::named_parameters)
//         .def("requires_grads_list", &Module::requires_grads_list)
//         .def("create_input", &Module::create_input)
//         .def("update_parameters", &Module::update_parameters)
//         .def("assert_parameters", &Module::assert_parameters)
//         .def("loss", &Module::loss)
//         .def("forward", &Module::forward);
// }
//
// }  // namespace TensorFrost