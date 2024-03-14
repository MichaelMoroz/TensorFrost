#include "Tensor.h"

namespace TensorFrost {

IR* Tensor::evaluation_context_ir_ = nullptr;

Node::~Node() { delete tensor_; }

void ArgumentManager::AddArgument(Arg* arg) {
	ArgID id = ArgID(arg->type_, arg->index_);
	arguments_[id] = arg->from_->node_;
	argument_types_[id] = arg->from_->node_->GetTensor()->type;
	argument_counts_[id.first]++;
}

int Node::TryComputeShape() {
	ArgMap shape = GetArgumentMap(ArgType::Shape);
	int size = 1;
	for (auto& [index, arg] : shape) {
		Node* shape_node = arg->from_->get();
		if (shape_node->name != "const") {
			return -1;  // can not compute shape at compile time
		}
		size *= arg->from_->get()->tensor_->data[0];
	}
	return size;
}

}  // namespace TensorFrost