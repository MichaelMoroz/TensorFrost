#include "IR.h"

namespace TensorFrost {

void IR::UpdateNodeOutputs() {
	for (auto node = begin(); !node.is_end(); ++node) {
		node->outputs_.clear();
	}

	for (auto node = begin(); !node.is_end(); ++node) {
		for (auto &input : node->inputs_) {
			input.from_->get()->outputs_.push_back(&input);
		}
	}
}

}