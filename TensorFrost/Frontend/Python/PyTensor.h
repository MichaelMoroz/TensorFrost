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
	~PyTensor() = default;

	[[nodiscard]] Tensor& Get() const { return *tensor_; }

	PyTensor(const std::vector<int>& shape, DataType type = DataType::Float) {
		switch (type) {
			case DataType::Float:
				tensor_ = &Tensor::Constant(shape, 0.0F);
				break;
			case DataType::Int:
				tensor_ = &Tensor::Constant(shape, 0);
				break;
			case DataType::Uint:
				tensor_ = &Tensor::Constant(shape, 0U);
				break;
			default:
				throw std::runtime_error("Invalid data type");
		}
	}

	PyTensor(const TensorView& indexed_tensor) {
		// load the elements of the indexed tensor
		tensor_ = &Tensor::Load(*indexed_tensor.value, indexed_tensor.indices);
	}

	PyTensor(float value) { tensor_ = &Tensor::Constant(Shape(), value); }
	PyTensor(int value) { tensor_ = &Tensor::Constant(Shape(), value); }
	PyTensor(unsigned int value) { tensor_ = &Tensor::Constant(Shape(), value); }
};

Tensor TensorFromPyArray(const py::array_t<float>& array);
py::array_t<float> TensorToPyArray(const Tensor& tensor);

}  // namespace TensorFrost