#include "IR.h"

namespace TensorFrost {

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

ScopeType GetScopeType(const Node* node) {
	// check if the node has a "kernel" operation parent
	// go over all parents and check if any of them is a kernel
	for (Node* parent = node->parent; parent != nullptr;
	     parent = parent->parent) {
		if (parent->name == "kernel") {
			return ScopeType::Kernel;
		} else if (parent->name == "host") {
			return ScopeType::Host;
		}
	}
	return ScopeType::None;
}

Node* Node::GetLastChild() {
	Node* last = nullptr;
	for (NodeIterator it = NodeIterator(this); !it.end(); it.next()) {
		last = it.get();
	}
	return last;
}

}  // namespace TensorFrost