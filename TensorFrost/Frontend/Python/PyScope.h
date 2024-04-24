
#pragma once

#include <TensorFrost.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace TensorFrost {

namespace py = pybind11;

class PyScope {
	PyTensor& tensor_;
public:
	PyScope(PyTensor& tensor) : tensor_(tensor) {}

	void __enter__() {
		tensor_.Get().Enter();
	}

	void __exit__(py::object exc_type, py::object exc_value,
	              py::object traceback) {
		tensor_.Get().Exit();
	}
};

}  // namespace TensorFrost
