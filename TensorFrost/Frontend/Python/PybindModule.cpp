#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

void PyTensorDefinition(py::module&, py::class_<PyTensor>&);
void TensorViewDefinition(py::module&, py::class_<TensorView>&);
void TensorFunctionsDefinition(py::module&);
void TensorProgramDefinition(py::module&, py::class_<TensorProgram>&);
void TensorMemoryDefinition(py::module& m, py::class_<TensorMemory>& py_tensor_mem);

PYBIND11_MODULE(TensorFrost, m) {
	auto data_type = py::enum_<DataType>(m, "DataType");
	auto py_tensor = py::class_<PyTensor>(m, "Tensor");
	auto tensor_view = py::class_<TensorView>(m, "TensorView");
	auto tensor_program = py::class_<TensorProgram>(m, "TensorProgram");
	auto py_tensor_mem = py::class_<TensorMemory>(m, "TensorMemory");

	data_type.value("float", DataType::Float);
	data_type.value("int", DataType::Int);
	data_type.value("uint", DataType::Uint);
	data_type.value("bool", DataType::Bool);
	m.attr("float32") = DataType::Float;
	m.attr("int32") = DataType::Int;
	m.attr("uint32") = DataType::Uint;
	m.attr("bool") = DataType::Bool;

	PyTensorDefinition(m, py_tensor);
	TensorViewDefinition(m, tensor_view);

	// implicit conversion from TensorView to PyTensor
	py::implicitly_convertible<TensorView, PyTensor>();
	py::implicitly_convertible<float, PyTensor>();
	py::implicitly_convertible<int, PyTensor>();
	py::implicitly_convertible<unsigned int, PyTensor>();

	TensorFunctionsDefinition(m);
	TensorProgramDefinition(m, tensor_program);
	TensorMemoryDefinition(m, py_tensor_mem);

	InitializeMemoryManager();

	py::print("TensorFrost module loaded!");
}

}  // namespace TensorFrost