#pragma once

#include <Tensor/Tensor.h>

namespace TensorFrost {
class TensorView {
 public:
	const Tensor* value;
	Tensors indices;
	Tensor* load;

	TensorView(const Tensor* value, Tensors& indices)
	    : value(value), indices(std::move(indices)) {
		load = &Tensor::Load(*this->value, this->indices);
	}
};
}  // namespace TensorFrost
