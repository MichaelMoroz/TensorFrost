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
	Tensor* tensor_;

 public:
	explicit PyTensor(Tensor* tensor) : tensor_(tensor) {}
	explicit PyTensor(const Tensor* tensor)
	    : tensor_(const_cast<Tensor*>(tensor)) {}
	~PyTensor() = default;

	[[nodiscard]] Tensor& Get() const { return *tensor_; }

	// PyTensor(const std::vector<int>& shape, DataType type = DataType::Float) {
	//	switch (type) {
	//		case DataType::Float:
	//			tensor_ = &Tensor::Constant(0.0F);
	//			break;
	//		case DataType::Int:
	//			tensor_ = &Tensor::Constant(shape, 0);
	//			break;
	//		case DataType::Uint:
	//			tensor_ = &Tensor::Constant(shape, 0U);
	//			break;
	//		default:
	//			throw std::runtime_error("Invalid data type");
	//	}
	// }

	explicit PyTensor(const TensorView& indexed_tensor) {
		// load the elements of the indexed tensor
		tensor_ = &Tensor::Load(*indexed_tensor.value, indexed_tensor.indices);
	}

	explicit PyTensor(float value) { tensor_ = &Tensor::Constant(value); }
	explicit PyTensor(int value) { tensor_ = &Tensor::Constant(value); }
	explicit PyTensor(unsigned int value) { tensor_ = &Tensor::Constant(value); }
};

using PyTensors = std::vector<PyTensor *>;

PyTensors PyTensorsFromTuple(const py::tuple& tuple);
Tensors TensorsFromTuple(const py::tuple& tuple);
PyTensors PyTensorsFromTensors(const Tensors& tensors);

}  // namespace TensorFrost