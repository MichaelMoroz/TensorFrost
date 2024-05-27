#include "Tensor.h"

namespace TensorFrost {

IR* Tensor::evaluation_context_ir_ = nullptr;

Node::~Node() { delete tensor_; }

vector<int> ShapeInfo::GetShape(int default_value) const {
	vector<int> shape;
	for (auto node: this->shape) {
		if(node->op->HasAllTypes(OpClass::Constant)) {
			shape.push_back(node->tensor_->TryGetConstant());
		} else {
			shape.push_back(default_value);
		}
	}
	return shape;
}

void ShapeInfo::ExpandDimensions(int new_dim)
{
	if(new_dim <= dim) {
		return;
	}
	Tensor& one = Tensor::Constant(1);
	for(int i = dim; i < new_dim; i++) {
	   InsertDim(0, one.node_);
	}
}

float ShapeInfo::GetSizeRatio(ShapeInfo& a, ShapeInfo& b) {
	vector<int> shape_a = a.GetShape();
	vector<int> shape_b = b.GetShape();
	float size_a = 1.0f;
	float size_b = 1.0f;
	for (int i = 0; i < shape_a.size(); i++) {
		size_a *= (float)shape_a[i];
	}
	for (int i = 0; i < shape_b.size(); i++) {
		size_b *= (float)shape_b[i];
	}
	return size_a / size_b;
}

void ArgumentManager::AddArgument(ArgID id, Node* node) {
	if(node == nullptr) {
		throw std::runtime_error("Node is null");
	}
	inputs_[id] = node;
	argument_types_[id] = node->GetTensor()->type;
	argument_counts_[id.first]++;
}

int Node::TryComputeShape() {
	Arguments shape = args.GetArguments(ArgType::Shape);
	int size = 1;
	for (auto& [index, shape_node] : shape) {
		if (shape_node->name != "const") {
			return -1;  // can not compute shape at compile time
		}
		size *= shape_node->tensor_->data[0];
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
	for (auto& [id, from] : other.node_->args.inputs_) {
		new_args[id] = from;
	}
	return GetCopy(other, new_args);
}

void Tensor::SetShape(Tensors shape) const {
	node_->args.RemoveArguments(ArgType::Shape);
	for (int i = 0; i < shape.size(); i++) {
		node_->args.AddArgument(ArgType::Shape, i, shape[i]->node_);
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

Tensor & Tensor::ReductionOP(string name, const Tensor &tensor, int axis, bool keepdims) {
	// get the shape of the tensor (all dimensions except the last one)
	Tensors shape = tensor.GetShape();
	axis = GetAxis((int)shape.size(), axis);

	//check if axis is valid
	if (axis < 0 || axis >= shape.size()) {
		throw std::runtime_error("Invalid axis for reduction operation " + name);
	}

	// remove the axis dimension
	shape.erase(shape.begin() + axis);
	if (shape.empty()) {
		shape.push_back(&Constant(1));
	}
	Tensor& op = OpShape(name, shape, &tensor);
	op.data = vector<uint>(1, axis);
	//if(keepdims) {
	//	op.node_->AddFlag(NodeFlag::KeepDims);
	//}
	return op;
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