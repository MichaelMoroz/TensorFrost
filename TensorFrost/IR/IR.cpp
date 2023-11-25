#include "IR.h"

namespace TensorFrost {

void IR::UpdateNodeOutputs() const {
	for (auto node = begin(); !node.is_end(); ++node) {
		node->outputs_.clear();
	}

	for (auto node = begin(); !node.is_end(); ++node) {
		for (auto& input : node->inputs_) {
			input.from_->get()->outputs_.push_back(&input);
		}
	}
}

void SwapLables(Node* a, Node* b) {
	// first swap the node addresses
	a->lable_->node_ = b;
	b->lable_->node_ = a;

	// now swap the labels
	Lable* temp = a->lable_;
	a->lable_ = b->lable_;
	b->lable_ = temp;
}

void CopyLable(Node* target, Node* copy) {
	// make old lable point to copy
	target->lable_->node_ = copy;
	// make new lable for target
	target->lable_ = new Lable(target);
}

}  // namespace TensorFrost