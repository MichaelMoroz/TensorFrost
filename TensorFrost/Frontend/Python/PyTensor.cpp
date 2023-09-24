#include "Frontend/Python/PyTensor.h"

namespace TensorFrost {

PyTensors PyTensorsFromTuple(const py::tuple& tuple) { 
	PyTensors tensors;
	for (auto arg : tuple) {
		tensors.push_back(&arg.cast<PyTensor&>());
	}
	return tensors;
}

Tensors TensorsFromTuple(const py::tuple& tuple) {
	Tensors tensors;
	for (auto arg : tuple) {
		tensors.push_back(&arg.cast<PyTensor&>().Get());
	}
	return tensors;
}

PyTensors PyTensorsFromTensors(const Tensors& tensors) { 
	PyTensors pyTensors;
	for (auto tensor : tensors) {
		pyTensors.push_back(new PyTensor(tensor));
	}
	return pyTensors;
}

}  // namespace TensorFrost
