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

}  // namespace TensorFrost