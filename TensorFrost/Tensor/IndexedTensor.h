#pragma once

#include <Tensor/Tensor.h>

namespace TensorFrost {
class TensorView {
 public:
	Tensor* value;
	std::vector<const Tensor*> indices;

	TensorView(Tensor* value, std::vector<const Tensor*> indices) {
		this->value = value;
		this->indices = std::move(indices);
	}
};
}  // namespace TensorFrost
