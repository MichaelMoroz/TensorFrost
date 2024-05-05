#include "Tensor.h"

namespace TensorFrost {

IR* Tensor::evaluation_context_ir_ = nullptr;

Node::~Node() { delete tensor_; }

float ShapeInfo::GetSizeRatio(ShapeInfo& a, ShapeInfo& b) {
	unordered_map<Node*, int> shape_map;
	for (auto& [index, node] : a.shape) {
		// if the node is a constant, use the constant value
		if (node->op->HasAllTypes(OpType::Constant)) {
			shape_map[node] = node->tensor_->TryGetConstant();
		} else { //just assume it equal to 256
			shape_map[node] = 256;
		}
	}
	for (auto& [index, node] : b.shape) {
		// if the node is a constant, use the constant value
		if (node->op->HasAllTypes(OpType::Constant)) {
			shape_map[node] = node->tensor_->TryGetConstant();
		} else { //just assume it equal to 256
			shape_map[node] = 256;
		}
	}
	
	float size_a = 1.0f;
	float size_b = 1.0f;

	for (auto& [index, node] : a.shape) {
		size_a *= (float)shape_map[node];
	}
	for (auto& [index, node] : b.shape) {
		size_b *= (float)shape_map[node];
	}

	return size_a / size_b;
}

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
	out.type = tensor.type;
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