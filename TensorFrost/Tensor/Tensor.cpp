#include "Tensor.h"

namespace TensorFrost {

IR* Tensor::evaluation_context_ir_ = nullptr;

Node::~Node() { delete tensor_; }

IR::~IR() {
	vector <Node*> to_delete;
	for (auto node = begin(); !node.is_end(); ++node) {
		to_delete.push_back(*node);
	}
	for (auto node : to_delete) {
		delete node;
	}

}

}  // namespace TensorFrost