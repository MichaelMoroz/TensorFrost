#include "Frontend/Python/PyTensor.h"

namespace TensorFrost {

Tensor TensorFromPyArray(const py::array_t<float>& array) {
	auto buffer = array.request();
	auto* ptr = static_cast<float*>(buffer.ptr);
	std::vector<int> shape = std::vector<int>();
	for (int i = 0; i < buffer.ndim; i++) {
		shape.push_back(buffer.shape[i]);
	}
	return Tensor::Constant(shape, ptr);
}

py::array_t<float> TensorToPyArray(const Tensor& tensor) {
	std::vector<int> shape = tensor.shape.GetShape();
	py::array::ShapeContainer shape2 =
	    py::array::ShapeContainer(shape.begin(), shape.end());
	py::array_t<float> array(shape2);
	auto buffer = array.request();
	auto* ptr = static_cast<float*>(buffer.ptr);
	for (int i = 0; i < tensor.Size(); i++) {
		ptr[i] = 0.0;
	}
	return array;
}

}  // namespace TensorFrost
