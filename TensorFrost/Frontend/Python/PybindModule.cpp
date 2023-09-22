#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

void PyTensorDefinition(py::module&, py::class_<PyTensor>&);
void TensorViewDefinition(py::module&, py::class_<TensorView>&);
void TensorFunctionsDefinition(py::module&);
void TensorProgramDefinition(py::module&, py::class_<TensorProgram>&);

PYBIND11_MODULE(TensorFrost, m) {
	auto dataType = py::enum_<TensorFrost::DataType>(m, "DataType");
	auto pyTensor = py::class_<PyTensor>(m, "Tensor");
	auto tensorView = py::class_<TensorView>(m, "TensorView");
	auto tensorProgram = py::class_<TensorProgram>(m, "TensorProgram");

	dataType.value("float", TensorFrost::DataType::Float);
	dataType.value("int", TensorFrost::DataType::Int);
	dataType.value("uint", TensorFrost::DataType::Uint);
	dataType.value("bool", TensorFrost::DataType::Bool);

	PyTensorDefinition(m, pyTensor);
	TensorViewDefinition(m, tensorView);

	// implicit conversion from TensorView to PyTensor
	py::implicitly_convertible<TensorView, PyTensor>();
	py::implicitly_convertible<float, PyTensor>();
	py::implicitly_convertible<int, PyTensor>();
	py::implicitly_convertible<unsigned int, PyTensor>();

	TensorFunctionsDefinition(m);
	TensorProgramDefinition(m, tensorProgram);

	py::print("TensorFrost module loaded!");
}

}  // namespace TensorFrost