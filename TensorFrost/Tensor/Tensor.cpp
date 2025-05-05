#include "Tensor.h"
#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {

Tensors Reverse(const Tensors& tensors) {
	Tensors reversed;
	for (int i = (int)tensors.size() - 1; i >= 0; i--) {
		reversed.push_back(tensors[i]);
	}
	return reversed;
}

vector<int> Reverse(const vector<int>& vec) {
	vector<int> reversed;
	for (int i = (int)vec.size() - 1; i >= 0; i--) {
		reversed.push_back(vec[i]);
	}
	return reversed;
}

int ReverseDim(int dim, size_t dims) {
	return (int)dims - dim - 1;
}

IR* Tensor::evaluation_context_ir_ = nullptr;

Node::~Node() { delete tensor_; }

std::string MakeNodeErrorMessage(std::string message, std::initializer_list<const Node*> nodes) {
	message += "\n";
	for (auto node : nodes) {
		message += GetNodeString(node) + "\n";
	}
	return message;
}

vector<int> ShapeInfo::GetShape(int default_value) const {
	vector<int> shape;
	for (auto [node, _]: this->shape) {
		if(node->op->class_ == OpClass::Constant) {
			shape.push_back(node->tensor_->TryGetConstant());
		} else {
			shape.push_back(default_value);
		}
	}
	return shape;
}

void ShapeInfo::ExpandDimensionsTo(int new_dim)
{
	if(new_dim <= dim) {
		return;
	}
	Tensor& one = Tensor::Constant(1);
	for(int i = dim; i < new_dim; i++) {
	   InsertDim(i, one.node_, true);
	}
}

float ShapeInfo::GetSizeEstimate(ShapeInfo& shape) {
	vector<int> shape_a = shape.GetShape();
	float size_a = 1.0f;
	for (int i = 0; i < shape_a.size(); i++) {
		size_a *= (float)shape_a[i];
	}
	return size_a;
}

void ArgumentManager::AddArgument(ArgID id, Node* node) {
	if(node == nullptr) {
		throw std::runtime_error("Node is null");
	}
	inputs_[id] = node;
	argument_types_[id] = node->format;
	argument_counts_[id.first]++;
	//add this node as an output of the argument
	node->args.AddOutput(id, node_);
}

void ArgumentManager::Remove(ArgID id) {
	if(inputs_.find(id) == inputs_.end()) {
		throw std::runtime_error("Cannot remove argument that does not exist");
	}
	//remove this node as an output of the argument
	inputs_[id]->args.RemoveOutput(id, node_);
	inputs_.erase(id);
	argument_types_.erase(id);
	argument_counts_[id.first]--;
}

void ArgumentManager::RemoveArguments(ArgType arg) {
	vector<ArgID> to_remove;
	for (auto& [id, node] : inputs_) {
		if (id.first == arg) {
			to_remove.push_back(id);
		}
	}
	for (auto& id : to_remove) {
		Remove(id);
	}
}

vector<const Tensor *> ArgumentManager::GetTensorVector(ArgType type) const  {
	vector<const Tensor*> tensors = vector<const Tensor*>(Count(type));
	for (auto& [id, node] : inputs_) {
		if (id.first == type) {
			tensors[id.second] = node->tensor_;
		}
	}
	return tensors;
}

tuple<const Operation *, TFDataFormat, ShapeInfo> Tensor::GetOperation(const string &name, const Tensors &tensors,
	bool check_shape) {
	vector<TFDataFormat> input_types = vector<TFDataFormat>();
	for (const auto& tensor : tensors) {
		input_types.push_back(tensor->node_->format);
	}

	const Operation* operation = FindOperation(name);

	// check if input is valid
	if (!operation->IsInputValid(input_types)) {
		string error = "Input types (";
		for (int i = 0; i < input_types.size(); i++) {
			error += DataTypeToString(input_types[i].type) + "(" + to_string(input_types[i].size) + ")";
			if (i < input_types.size() - 1) {
				error += ", ";
			}
		}
		error += ") are not valid for operation \"" + name + "\"";
		throw std::runtime_error(error);
	}

	ShapeInfo shape_info = ShapeInfo();

	if (check_shape)
	{
		//check if shapes are compatible and get the final broadcasted shape
		for (int i = 0; i < tensors.size(); i++) {
			ShapeInfo shape_info2 = ShapeInfo(tensors[i]->node_);
			auto result = CompareShape(shape_info, shape_info2, true);
			shape_info = result.broadcast_shape;
		}
	}

	TFDataFormat output_type = operation->GetOutputType(input_types);

	return {operation, output_type, shape_info};
}

bool Tensor::CheckIndices(const Tensors &indices) {
	for (const Tensor* index : indices) {
		if (index->node_->format.type != TFType::Int && index->node_->format.type != TFType::Uint) {
			return false;
		}
	}
	return true;
}

TFDataFormat Tensor::GetFormat() const { return node_->format; }

void Tensor::SetData(const vector<uint> &data) const {
	node_->data = data;
}

void Tensor::SetData(uint data) const {
	SetData(vector<uint>(1, data));
}

void Tensor::SetData(float data) const {
	SetData(vector<uint>(1, AsUint(data)));
}

void Tensor::SetData(int data) const {
	SetData(vector<uint>(1, AsUint(data)));
}

void Tensor::SetFormat(TFDataFormat type) const {
	node_->format = type;
}

void Tensor::DetachGrad() const {
	node_->flags.set(NodeProp::DetachGrad);
}

void Tensor::PassGrad() const {
	node_->flags.set(NodeProp::PassGrad);
}

void Tensor::StopFusion() const {
	node_->flags.set(NodeProp::StopFusion);
}

void Tensor::HintRange(float min, float max) const {
	node_->flags.set(NodeProp::HintMinValue, (int64_t)AsUint(min));
	node_->flags.set(NodeProp::HintMaxValue, (int64_t)AsUint(max));
}

void Tensor::HintRange(int min, int max) const {
	node_->flags.set(NodeProp::HintMinValue, (int64_t)min);
	node_->flags.set(NodeProp::HintMaxValue, (int64_t)max);
}

void Tensor::HintRange(uint min, uint max) const {
	node_->flags.set(NodeProp::HintMinValue, (int64_t)min);
	node_->flags.set(NodeProp::HintMaxValue, (int64_t)max);
}

Tensor* Tensor::GetCopy(const Tensor& other, NodeArguments args) {
	Tensor* copy = &CreateNode(other.node_->format, std::move(args), other.node_->name);
	copy->node_->data = other.node_->data;
	copy->node_->CopyProperties(other.node_);
	return copy;
}

Tensor* Tensor::GetCopy(const Tensor& other) {
	NodeArguments new_args;
	for (auto& [id, from] : other.node_->args.Inputs()) {
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

Tensors Tensor::GetInputShapeTensors(Tensors shape) {
	Tensors result = Tensors();
	for (int dim = 0; dim < shape.size(); dim++) {
		const Tensor* tensor = shape[dim];
		//check if tensor is a negative constant
		if (tensor->node_->name == "const" && (*(int*)&(tensor->node_->data[0])) < 0)
		{
			Tensor& mem = Static("input_shape", {TFType::Int, 32});
			//make sure its reversed on the backend
			mem.node_->flags.set(NodeProp::InputShapeDim, (int64_t)(shape.size() - dim - 1));
			result.push_back(&mem);
		}
		else
		{
			result.push_back(tensor);
		}
	}
	return result;
}

//Get values from a tensor at the given indices
Tensor& Tensor::Load(const Tensor& tensor, const Tensors& indices, IndexingMode mode) {
	Tensor& out = MemoryOp("load", &tensor, indices);
	out.node_->indexing_mode_ = mode;
	out.SetData(0);
	out.SetDebugName(tensor.node_->debug_name);
	return out;
}

Tensor& Tensor::Store(const Tensor& tensor, const Tensor& value,
                      const Tensors& indices, IndexingMode mode) {
	Tensor& out = MemoryOp("store", &tensor, indices, &value);
	out.node_->indexing_mode_ = mode;
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
	op.node_->data = vector<uint>(1, axis);
	//if(keepdims) {
	//	op.node_->AddFlag(NodeFlag::KeepDims);
	//}
	return op;
}

Tensor & Tensor::ScanOP(string name, const Tensor &tensor, int axis) {
	Tensor& op = Op(name, &tensor);
	op.node_->data = vector<uint>(1, axis);
	return op;
}

bool Tensor::AreTensorsEqual(const Tensor &a, const Tensor &b) {
	if(a.node_->op->class_ == OpClass::Constant  && b.node_->op->class_ == OpClass::Constant) {
		return a.node_->data[0] == b.node_->data[0];
	}
	if(a.node_ == b.node_) {
		return true;
	}
	return false;
}

Tensor& Tensor::Reshape(const Tensor& tensor, const Tensors& shape, TFDataFormat format) {
	Tensor& out = MemoryOpShape("reshape", shape, &tensor);
	out.SetDebugName(tensor.node_->debug_name);
	if(format.type != TFType::None) {
		out.node_->format = format;
	} else {
		out.node_->format = tensor.node_->format;
	}
	return out;
}

Tensor & Tensor::Assert(const Tensor &tensor, const Tensors &shape, TFDataFormat type) {
	Tensor& out = MemoryOpShape("assert", shape, &tensor);
	out.SetDebugName(tensor.node_->debug_name);
	out.node_->format = type;
	return out;
}

void Tensor::SetDebugName(const string& name) const
{
	if (name != "") {
		node_->debug_name = name;
	}

	if (strip_debug_names) {
		static int count = 0;

		node_->debug_name = "tensor_" + std::to_string(count++);
	}
}

void Tensor::BeginRegion(const string& name) {
	Tensor& t = Static("region_begin", {TFType::None, 0});
	t.SetDebugName(name);
}

void Tensor::EndRegion(const string& name) {
	Tensor& t = Static("region_end", {TFType::None, 0});
	t.SetDebugName(name);
}

const Tensor* Node::GetTensor() const {
	if (tensor_->node_ != this) {
		throw std::runtime_error("Fatal Error: Tensor node does not match");
	}
	return tensor_;
}


}  // namespace TensorFrost