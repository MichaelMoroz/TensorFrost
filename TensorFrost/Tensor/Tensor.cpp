#include "Tensor.h"

namespace TensorFrost {

IR* Tensor::evaluation_context_ir_ = nullptr;

Node::~Node() { delete tensor_; }

IR::~IR() {
	for (auto& node : nodes_) delete node;
}

}  // namespace TensorFrost