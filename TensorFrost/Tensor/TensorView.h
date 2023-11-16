#pragma once

#include <Tensor/Tensor.h>

namespace TensorFrost {
class TensorView {
 public:
	const Tensor* value;
	Tensors indices;

	TensorView(const Tensor* value, Tensors& indices)
	    : value(value), indices(std::move(indices)) {}
};
}  // namespace TensorFrost
