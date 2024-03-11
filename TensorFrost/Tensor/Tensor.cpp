#include "Tensor.h"

namespace TensorFrost {

IR* Tensor::evaluation_context_ir_ = nullptr;

Node::~Node() { delete tensor_; }

ArgumentTypes Node::GetArgumentTypes()
{
	ArgumentTypes result = ArgumentTypes();
	for (auto& input : inputs_) {
		result[ArgID(input.type_, input.index_)] =
		    input.from_->node_->GetTensor()->type;
	}
	return result;
}

}  // namespace TensorFrost