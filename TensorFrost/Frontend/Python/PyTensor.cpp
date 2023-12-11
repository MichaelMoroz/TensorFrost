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

PyTensors PyTensorsFromList(const py::list& list) {
	PyTensors tensors;
	for (auto arg : list) {
		tensors.push_back(&arg.cast<PyTensor&>());
	}
	return tensors;
}

Tensors TensorsFromList(const py::list& list) {
	Tensors tensors;
	for (auto arg : list) {
		tensors.push_back(&arg.cast<PyTensor&>().Get());
	}
	return tensors;
}

vector<TensorMemory*> TensorMemoryFromTuple(const py::tuple& tuple) {
	vector<TensorMemory*> memories;
	for (auto arg : tuple) {
		memories.push_back(&arg.cast<TensorMemory&>());
	}
	return memories;
}

vector<TensorMemory*> TensorMemoryFromList(const py::list& list) {
	vector<TensorMemory*> memories;
	for (auto arg : list) {
		memories.push_back(&arg.cast<TensorMemory&>());
	}
	return memories;
}

PyTensors PyTensorsFromTensors(const Tensors& tensors) {
	PyTensors py_tensors;
	for (const auto* tensor : tensors) {
		py_tensors.push_back(new PyTensor(tensor));
	}
	return py_tensors;
}

std::string r_op(const std::string& name) { return "__r" + name + "__"; }

std::string l_op(const std::string& name) { return "__" + name + "__"; }

}  // namespace TensorFrost
