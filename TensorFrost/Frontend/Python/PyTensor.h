#pragma once

#include <TensorFrost.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace TensorFrost {

#define PT(tensor) PyTensor(&(tensor))
#define T(tensor) (tensor).Get()

namespace py = pybind11;

// Tensor wrapper for python
class PyTensor {
	const Tensor* tensor_;

 public:
	explicit PyTensor(Tensor* tensor) : tensor_(tensor) {}
	explicit PyTensor(const Tensor* tensor) : tensor_(tensor) {}
	~PyTensor() = default;

	[[nodiscard]] const Tensor& Get() const { return *tensor_; }

	explicit PyTensor(const TensorView& indexed_tensor) {
		// load the elements of the indexed tensor
		tensor_ = &Tensor::Load(*indexed_tensor.value, indexed_tensor.indices);
	}

	explicit PyTensor(float value) { tensor_ = &Tensor::Constant(value); }
	explicit PyTensor(int value) { tensor_ = &Tensor::Constant(value); }
	explicit PyTensor(unsigned int value) { tensor_ = &Tensor::Constant(value); }
};

using PyTensors = std::vector<PyTensor*>;

PyTensors PyTensorsFromTuple(const py::tuple& tuple);
Tensors TensorsFromTuple(const py::tuple& tuple);
PyTensors PyTensorsFromList(const py::list& list);
Tensors TensorsFromList(const py::list& list);
PyTensors PyTensorsFromTensors(const Tensors& tensors);

vector<TensorMemory*> TensorMemoryFromTuple(const py::tuple& tuple);
vector<TensorMemory*> TensorMemoryFromList(const py::list& list);

std::string r_op(const std::string& name);
std::string l_op(const std::string& name);

}  // namespace TensorFrost