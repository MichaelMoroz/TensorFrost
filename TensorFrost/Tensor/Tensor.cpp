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

Tensor* Tensor::GetCopy(const Tensor& other, Arguments args) {
	Tensor* copy = &CreateNode(other.type, std::move(args), other.node_->name);
	copy->data = other.data;
	copy->node_->CopyProperties(other.node_);
	return copy;
}

Tensor* Tensor::GetCopy(const Tensor& other) {
	Arguments new_args;
	for (auto& arg : other.node_->inputs_) {
		new_args.push_back(arg);
	}
	return GetCopy(other, new_args);
}

void Tensor::SetShape(Tensors shape) const {
	node_->RemoveArguments(ArgType::Shape);
	for (int i = 0; i < shape.size(); i++) {
		node_->AddArgument(shape[i]->node_, ArgType::Shape, i);
	}
}

//Get values from a tensor at the given indices
Tensor& Tensor::Load(const Tensor& tensor, const Tensors& indices,
                     bool unsafe) {
	Tensor& out = MemoryOp("load", &tensor, indices);
	if (unsafe) out.node_->indexing_mode_ = TensorIndexingMode::Unsafe;
	out.SetDebugName(tensor.node_->debug_name);
	return out;
}

Tensor& Tensor::Store(const Tensor& tensor, const Tensor& value,
                      const Tensors& indices, bool unsafe) {
	Tensor& out = MemoryOp("store", &tensor, indices, &value);
	if (unsafe) out.node_->indexing_mode_ = TensorIndexingMode::Unsafe;
	return out;
}

Tensor& Tensor::Reshape(const Tensor& tensor, const Tensors& shape) {
	Tensor& out = MemoryOpShape("reshape", shape, &tensor);
	out.SetDebugName(tensor.node_->debug_name);
	return out;
}

void Tensor::SetDebugName(const string& name) const
{
	if (name != "")
	{
		node_->debug_name = name;
	}
}

}  // namespace TensorFrost