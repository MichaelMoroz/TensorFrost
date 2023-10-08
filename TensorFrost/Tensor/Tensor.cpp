#include "Tensor.h"

namespace TensorFrost {

IR* Tensor::evaluation_context_ir_ = nullptr;

IR::~IR() {
	for (Tensor* node : nodes_) {
		delete node;
	}
}

}  // namespace TensorFrost