#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

void PyTensorDefinition(py::module&, py::class_<PyTensor>&);
void TensorViewDefinition(py::module&, py::class_<TensorView>&);
void TensorFunctionsDefinition(py::module&);
void TensorProgramDefinition(py::module&, py::class_<TensorProgram>&);

PYBIND11_MODULE(TensorFrost, m) {
	auto data_type = py::enum_<DataType>(m, "DataType");
	auto py_tensor = py::class_<PyTensor>(m, "Tensor");
	auto tensor_view = py::class_<TensorView>(m, "TensorView");
	auto tensor_program = py::class_<TensorProgram>(m, "TensorProgram");

	data_type.value("float", DataType::Float);
	data_type.value("int", DataType::Int);
	data_type.value("uint", DataType::Uint);
	data_type.value("bool", DataType::Bool);

	PyTensorDefinition(m, py_tensor);
	TensorViewDefinition(m, tensor_view);

	// implicit conversion from TensorView to PyTensor
	py::implicitly_convertible<TensorView, PyTensor>();
	py::implicitly_convertible<float, PyTensor>();
	py::implicitly_convertible<int, PyTensor>();
	py::implicitly_convertible<unsigned int, PyTensor>();

	TensorFunctionsDefinition(m);
	TensorProgramDefinition(m, tensor_program);

	py::print("TensorFrost module loaded!");
}

}  // namespace TensorFrost