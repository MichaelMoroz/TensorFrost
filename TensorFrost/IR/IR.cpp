#include "IR.h"

namespace TensorFrost {

void IR::UpdateNodeOutputs() const {
	for (auto node = begin(); !node.is_end(); ++node) {
		node->outputs_.clear();
		node->UpdateOutputs();
	}
}

int MaxIndexCount(ArgMap& map) {
	if (map.empty()) return 0;
	// maps are sorted by index, so the last element has the highest index
	return map.rbegin()->first + 1;
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