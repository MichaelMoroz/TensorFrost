#pragma once

#include <Tensor/Tensor.h>

namespace TensorFrost {
class TensorView {
 public:
	const Tensor* value;
	std::vector<const Tensor*> indices;

	TensorView(const Tensor* value, std::vector<const Tensor*>&& indices)
	    : value(value), indices(std::move(indices)) {}
};
}  // namespace TensorFrost
