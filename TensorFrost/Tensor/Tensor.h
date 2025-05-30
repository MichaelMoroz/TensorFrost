#pragma once

#include <algorithm>
#include <list>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <variant>

// for FLT_MAX, INT_MAX, etc.
#include <float.h>
#include <limits.h>

#include <math.h>

#include "Compiler/Graph/IR.h"
#include "Utility/Utility.h"

namespace TensorFrost {

using Tensors = vector<const Tensor*>;

Tensors Reverse(const Tensors& tensors);
vector<int> Reverse(const vector<int>& vec);
int ReverseDim(int dim, size_t dims);

class Tensor {
 private:
	static IR* evaluation_context_ir_;

	static Tensor& CreateNode(TFDataFormat type, NodeArguments args, string name) {
		if (evaluation_context_ir_ == nullptr) {
			throw std::runtime_error(
			    "Evaluation context has not been set. Are you doing operations "
			    "without compiling first?");
		}

		//TODO: merge Tensor and Node classes
		auto* tensor = new Tensor();
		tensor->node_ = evaluation_context_ir_->AddNode(tensor, std::move(args), std::move(name), type);
		return *tensor;
	}

	static void AddArgument(NodeArguments& arguments, const Tensor* tensor,
	                        ArgType type, int index = 0) {
		arguments[ArgID(type, index)] = tensor->node_;
	}

	static void AddArguments(NodeArguments& arguments, const Tensors& tensors,
	                         ArgType type) {
		for (int i = 0; i < tensors.size(); i++) {
			AddArgument(arguments, tensors[i], type, i);
		}
	}

	static void AddArguments(NodeArguments& arguments, const NodeArguments& toadd) {
		for (const auto& i : toadd) {
			arguments[i.first] = i.second;
		}
	}

	static bool AssertTensorShape(const Tensor* a, const Tensor* b, bool throw_error = true) {
		return CompareShape(a->node_, b->node_, throw_error).compatible;
	}

	static tuple<const Operation*, TFDataFormat, ShapeInfo> GetOperation(const string& name,
	                                              const Tensors& tensors, bool check_shape = true);

	template <typename... Args>
	static Tensor& Op(std::string op, const Args*... args) {
		op = RemoveSpaces(op);

		if (op.empty()) {
			throw std::runtime_error("Operation name cannot be empty");
		}

		// convert the parameter pack to a std::vector
		Tensors tensors = {args...};

		// get the operation and output type
		auto [operation, output_type, shape_info] = GetOperation(op, tensors);

		// create argument list
		NodeArguments arguments = NodeArguments();

		AddArguments(arguments, tensors, ArgType::Input);

		AddArguments(arguments, shape_info.GetArguments());

		return CreateNode(output_type, arguments, op);
	}

public:
	static Tensor& OpShape(std::string op, Tensors shape, Tensors tensors) {
		op = RemoveSpaces(op);

		if (op.empty()) {
			throw std::runtime_error("Operation name cannot be empty");
		}

		// get the operation and output type
		auto [operation, output_type, shape_info] = GetOperation(op, tensors, false);

		// create argument list
		NodeArguments arguments = NodeArguments();

		AddArguments(arguments, tensors, ArgType::Input);
		AddArguments(arguments, shape, ArgType::Shape);

		return CreateNode(output_type, arguments, op);
	}

	template <typename... Args>
	static Tensor& OpShape(std::string op, Tensors shape, const Args*... args) {
		// convert the parameter pack to a std::vector
		Tensors tensors = {args...};

		return OpShape(op, shape, tensors);
	}

	template <typename... Args>
	static Tensor& MemoryOpShape(std::string op, Tensors shape, const Tensor* memory, const Args*... args) {
		op = RemoveSpaces(op);

		if (op.empty()) {
			throw std::runtime_error("Memory operation name cannot be empty");
		}

		// convert the parameter pack to a std::vector
		Tensors tensors = {args...};

		// get the operation and output type
		auto [operation, output_type, shape_info] = GetOperation(op, tensors);

		// create argument list
		NodeArguments arguments = NodeArguments();

		AddArgument(arguments, memory, ArgType::Memory);
		AddArguments(arguments, tensors, ArgType::Input);
		AddArguments(arguments, shape, ArgType::Shape);

		return CreateNode(output_type, arguments, op);
	}

	static bool CheckIndices(const Tensors& indices);

	template <typename... Args>
	static Tensor& MemoryOp(string op, const Tensor* memory,
	                        const Tensors indices, const Args*... args) {
		op = RemoveSpaces(op);

		if (op.empty()) {
			throw std::runtime_error("Memory operation name cannot be empty");
		}

		if(!CheckIndices(indices)) {
			throw std::runtime_error("Tensor indices must be integers");
		}

		//check if memory op is local
		bool is_local = false;
		if(memory->node_->op->HasAllTypes(OpProp::LocalMemory)) {
			is_local = true;
		}

		//if local, can only have 1 index
		if(is_local && indices.size() > 1) {
			throw std::runtime_error("Local memory operations can only have 1 index");
		}

		//if global, can only have up to memory dimension indices
		size_t memory_dim = memory->GetDimension();
		if(!is_local && indices.size() > memory_dim) {
			throw std::runtime_error("Too many indices for memory operation, memory has " + std::to_string(memory_dim) + " dimensions, while " + std::to_string(indices.size()) + " indices were provided");
		}

		// convert the parameter pack to a std::vector
		Tensors tensors = {args...};

		// get the operation and output type
		auto [operation, output_type, shape_info] = GetOperation(op, tensors);

		if (operation->HasAllTypes(OpProp::Modifier))
		{
			memory->node_->flags.set(NodeProp::Modified);
		}

		// create argument list
		NodeArguments arguments = NodeArguments();

		AddArgument(arguments, memory, ArgType::Memory);
		AddArguments(arguments, tensors, ArgType::Input);
		AddArguments(arguments, indices, ArgType::Index);

		// get an input node that has shape arguments
		NodeArguments shape_arguments = shape_info.GetArguments();

		//use index shape instead if no input shape is found
		if (shape_arguments.empty())
		{
			for (const Tensor* index : indices)
			{
				shape_arguments = index->node_->args.GetArguments(ArgType::Shape);
				if (!shape_arguments.empty()) {
					break;
				}
			}
		}

		//if no indices or inputs exist, use memory shape
		if (indices.empty() && tensors.empty())
		{
			shape_arguments = memory->node_->args.GetArguments(ArgType::Shape);
		}

		AddArguments(arguments, shape_arguments);

		if (op == "load") output_type = memory->GetFormat();

		Tensor& output = CreateNode(output_type, arguments, op);
		if(is_local) output.node_->flags.set(NodeProp::LocalMemoryOp);
		return output;
	}

	static Tensor& Static(string op, const NodeArguments& shape,
	                      const TFDataFormat format) {
		op = RemoveSpaces(op);

		if (op.empty()) {
			throw std::runtime_error("Static operation name cannot be empty");
		}

		const Operation* operation = FindOperation(op);
		// check if output is valid
		if (!operation->IsOutputValid(format)) {
			throw std::runtime_error("Type " + DataTypeToString(format.type) + "(" + std::to_string(format.size) + ") is not valid for operation \"" + op + "\"");
		}
		NodeArguments arguments = NodeArguments();
		AddArguments(arguments, shape);
		return CreateNode(format, arguments, op);
	}

	static Tensor& Static(const string& op, const Tensors& shape,
	                      const TFDataFormat type) {
		NodeArguments arguments = NodeArguments();
		AddArguments(arguments, shape, ArgType::Shape);
		return Static(op, arguments, type);
	}

	static Tensor& Static(const string& op, const TFDataFormat type) {
		return Static(op, NodeArguments(), type);
	}

	static void SetEvaluationContext(IR* ir) {
		if (evaluation_context_ir_ != nullptr && ir != nullptr) {
			throw std::runtime_error("Evaluation context change is forbidden.");
		}
		evaluation_context_ir_ = ir;
	}

	static IR* GetEvaluationContext() {
		return evaluation_context_ir_;
	}

	string GetConstantString() const;

	static Tensor& CustomOperation(const string & name, Tensors inputs, Tensors shape) {
		return OpShape(name, shape, inputs);
	}

	Node* node_ = nullptr;

	TFDataFormat GetFormat() const;
	void SetData(const vector<uint>& data) const;
	void SetData(uint data) const;
	void SetData(float data) const;
	void SetData(int data) const;
	void SetFormat(TFDataFormat type) const;
	void DetachGrad() const;
	void PassGrad() const;
	void StopFusion() const;
	void HintRange(float min, float max) const;
	void HintRange(int min, int max) const;
	void HintRange(uint min, uint max) const;

	static Tensor* GetCopy(const Tensor& other, NodeArguments args);

	static Tensor* GetCopy(const Tensor& other);

	void SetMemoryType(NodeProp memory_type, int index = 0) const {
		node_->SetMemoryType(memory_type, index);
	}

	int GetDimension() const {
		ShapeInfo shape_info = ShapeInfo(node_);
		return shape_info.dim;
	}

	Tensors GetShape() const {
		ShapeInfo shape_info = ShapeInfo(node_);
		return shape_info.GetTensors();
	}

	Tensors GetReverseShape() const {
		Tensors shape = GetShape();
		std::reverse(shape.begin(), shape.end());
		return shape;
	}

	ShapeInfo GetShapeInfo() const {
		return ShapeInfo(node_);
	}

	void SetShape(Tensors shape) const;

	int TryGetConstant() const {
		if (node_->name != "const") {
			return -1;
		}
		return AsInt(node_->data[0]);
	}

	// tensor factory methods
	static Tensors GetConstantShape(const vector<int>& shape) {
		Tensors result = Tensors();
		for (int i : shape) {
			result.push_back(&Constant(i));
		}
		return result;
	}

	static Tensor& Constant(float value) {
		Tensor& output = Static("const", TFTypeFloat32);
		output.SetData(AsUint(value));
		return output;
	}
	static Tensor& Constant(int value) {
		Tensor& output = Static("const", TFTypeInt32);
		output.SetData(AsUint(value));
		return output;
	}
	static Tensor& Constant(uint value) {
		Tensor& output = Static("const", TFTypeUint32);
		output.SetData(value);
		return output;
	}
	static Tensor& Constant(bool value) {
		Tensor& output = Static("const", TFTypeBool32);
		output.SetData(value);
		return output;
	}
	static Tensor& Constant(uint value, TFDataFormat type) {
		Tensor& output = Static("const", type);
		output.SetData(value);
		return output;
	}
	static Tensor& Constant(uint value, const Tensors& shape, TFDataFormat type) {
		NodeArguments arguments = NodeArguments();
		AddArguments(arguments, shape, ArgType::Shape);
		Tensor& output = Static("const", arguments, type);
		output.SetData(value);
		return output;
	}

	static Tensor& Constant(const Tensors& shape, float value) {
		NodeArguments arguments = NodeArguments();
		AddArguments(arguments, shape, ArgType::Shape);
		Tensor& output = Static("const", arguments, TFTypeFloat32);
		output.SetData(value);
		return output;
	}
	static Tensor& Constant(const vector<int>& shape, float value) {
		return Constant(GetConstantShape(shape), value);
	}
	static Tensor& Constant(const Tensors& shape, int value) {
		NodeArguments arguments = NodeArguments();
		AddArguments(arguments, shape, ArgType::Shape);
		Tensor& output = Static("const", arguments, TFTypeInt32);
		output.SetData(value);
		return output;
	}
	static Tensor& Constant(const vector<int>& shape, int value) {
		return Constant(GetConstantShape(shape), value);
	}

	static Tensor& Constant(const Tensors& shape, uint value) {
		NodeArguments arguments = NodeArguments();
		AddArguments(arguments, shape, ArgType::Shape);
		Tensor& output = Static("const", arguments, TFTypeUint32);
		output.SetData(value);
		return output;
	}

	static Tensor& Constant(const vector<int>& shape, uint value) {
		return Constant(GetConstantShape(shape), value);
	}
	static Tensor& Constant(const Tensors shape, uint value, TFDataFormat type) {
		NodeArguments arguments = NodeArguments();
		AddArguments(arguments, shape, ArgType::Shape);
		Tensor& output = Static("const", arguments, type);
		output.SetData(value);
		return output;
	}

	static Tensors GetShapeTensors(const vector<int>& shape) {
		Tensors result = Tensors();
		for (int i : shape) {
			result.push_back(&Constant(i));
		}
		return result;
	}

	static Tensor& Memory(const TFDataFormat type) { return Static("memory", type); }
	static Tensor& Memory(const Tensors& shape,
	                      const TFDataFormat type = TFTypeFloat32) {
		return Static("memory", shape, type);
	}
	static Tensor& Memory(const NodeArguments& shape,
	                      const TFDataFormat type = TFTypeFloat32) {
		return Static("memory", shape, type);
	}
	static Tensor& Memory(const vector<int>& shape,
		const TFDataFormat type = TFTypeFloat32) {
		return Memory(GetShapeTensors(shape), type);
	}

	static Tensor& LocalMemory(const int size, const TFDataFormat type) {
		Tensor& output = Static("local_memory", type);
		output.SetData(size);
		return output;
	}

	static Tensor& GroupMemory(const int size, const TFDataFormat type) {
		Tensor& output = Static("group_memory", type);
		output.SetData(size);
		return output;
	}

	static void GroupBarrier() {
		Op("group_barrier");
	}

	static Tensors GetInputShapeTensors(Tensors shape);

	static Tensor& Input(const TFDataFormat type = TFTypeFloat32) {
		Tensor& output = Memory(type);
		output.SetMemoryType(NodeProp::InputMemory);
		return output;
	}
	static Tensor& Input(const Tensors& shape,
	                     const TFDataFormat type = TFTypeFloat32) {
		Tensor& output = Memory(GetInputShapeTensors(shape), type);
		output.SetMemoryType(NodeProp::InputMemory);
		return output;
	}
	static Tensor& Input(const vector<int>& shape,
	                     const TFDataFormat type = TFTypeFloat32) {
		return Input(GetShapeTensors(shape), type);
	}

	static Tensor& Index(NodeArguments shape, int dim) {
		Tensor& output = Static("dim_id", shape, TFTypeInt32);
		output.SetData(dim);
		output.SetFormat(TFTypeInt32);
		return output;
	}

	static Tensor& Index(Tensors shape, int dim) {
		Tensor& output = Static("dim_id", shape, TFTypeInt32);
		output.SetData(dim);
		output.SetFormat(TFTypeInt32);
		return output;
	}

	static Tensor& Index(const vector<int>& shape, int dim) {
		return Index(GetConstantShape(shape), dim);
	}

	static Tensors Indices(Tensors shape) {
		int dims = (int)shape.size();
		Tensors indices = Tensors();
		for (int i = 0; i < dims; i++) {
			indices.push_back(&Index(shape, i));
		}
		return indices;
	}

	static Tensor& FlatIndex(Tensors shape, Tensors indices) {
		int memory_dim = (int)shape.size();
		if(memory_dim == 0) return Constant(0);
		// compute the flat index (C-order)
		Tensor* flat_index = const_cast<Tensor*>(indices[0]);
		for (int i = 1; i < memory_dim; i++) {
			flat_index = &(*flat_index * *shape[i]);
			flat_index = &(*flat_index + *indices[i]);
		}
		return *flat_index;
	}

	static Tensors IndicesFromFlatIndex(const Tensor* index, Tensors shape)
	{
		size_t dims = shape.size();
		Tensors indices = Tensors(dims);
		Tensors sizes = Tensors(dims);
		sizes[0] = shape[0];
		for (size_t i = 1; i < dims - 1; i++) {
			sizes[i] = &(*sizes[i - 1] * *shape[i]);
		}

		Tensor* temp;
		for (size_t i = 0; i < dims; i++) {
			Tensor* idx0 = const_cast<Tensor*>(index);
			if (i < dims - 1) {
				idx0 = &(*idx0 / *sizes[dims - i - 2]);
			}
			if (i > 0) {
				temp = &(*temp * *shape[dims - i - 1]);
				idx0 = &(*idx0 - *temp);
				if (i != dims - 1) temp = &(*temp + *idx0);
			} else {
				temp = idx0;
			}
			indices[dims - i - 1] = idx0;
		}

		return indices;
	}

	static Tensor& ElementIndex(Tensors shape) {
		return FlatIndex(shape, Indices(shape));
	}

	static Tensor& GetSeed(Tensors shape, const Tensor& seed) {
		Tensor* full_seed = &const_cast<Tensor&>(seed);
		if(full_seed->node_->format.type != TFType::Uint) {
			full_seed = &asuint(*full_seed); //convert seed to uint
		}
		full_seed = &(touint(ElementIndex(shape)) + *full_seed * Constant(2654435761u));
		return *full_seed;
	}

	static Tensor& Hash(Tensors shape, const Tensor& seed) {
		return pcg(GetSeed(shape, seed));
	}

	static Tensor& Random(Tensors shape, const Tensor& seed) {
		return pcgf(GetSeed(shape, seed));
	}

	Tensor& BlockIndex() const {
		Tensor& output = Static(
		    "block_id", node_->args.GetArguments(ArgType::Shape), TFTypeInt32);
		output.SetFormat(TFTypeInt32);
		return output;
	}

	Tensor& BlockThreadIndex(int i) const {
		Tensor& output = Static(
		    "block_thread_id", node_->args.GetArguments(ArgType::Shape), TFTypeInt32);
		output.SetFormat(TFTypeInt32);
		output.SetData(i);
		return output;
	}

	static Tensor& Load(const Tensor& tensor, const Tensors& indices = Tensors(),
	                    IndexingMode mode = IndexingMode::Clamp);

	static Tensor& Deallocate(const Tensor& tensor) {
		return MemoryOp("deallocate", &tensor, {});
	}

	Tensor& Index(int dim) const {
		Tensor& output = Static("dim_id", node_->args.GetArguments(ArgType::Shape), TFTypeInt32);
		output.SetData(dim);
		output.SetFormat(TFTypeInt32);
		return output;
	}

	Tensors Indices() const {
		return Indices(GetShape());
	}

	static Tensor& Store(const Tensor& tensor, const Tensor& value,
	                     const Tensors& indices = Tensors(), IndexingMode mode = IndexingMode::Clamp);

	void Set(const Tensor& value) const  {
		//check if memory and value shapes are compatible
		ShapeCompareResult shape_result = CompareShape(node_, value.node_, true);
		if (!shape_result.compatible) {
            throw std::runtime_error("Cannot set tensor with incompatible shape");
        }
		MemoryOp("set", this, {}, &value);
		//update the shape of the tensor
		SetShape(shape_result.broadcast_shape.GetTensors());
	}

	static void ScatterAdd(const Tensor& tensor, const Tensor& value,
	                       const Tensors& indices, IndexingMode mode = IndexingMode::Clamp) {
		MemoryOp("InterlockedAdd", &tensor, indices, &value);
	}

	static Tensor& ScatterAddPrev(const Tensor& tensor, const Tensor& value,
		const Tensors& indices, IndexingMode mode = IndexingMode::Clamp) {
		Tensor& a = MemoryOp("InterlockedAdd_Prev", &tensor, indices, &value);
		a.node_->indexing_mode_ = mode;
		return a;
	}

	static void ScatterMax(const Tensor& tensor, const Tensor& value,
	                       const Tensors& indices, IndexingMode mode = IndexingMode::Clamp) {
		Tensor& a = MemoryOp("InterlockedMax", &tensor, indices, &value);
		a.node_->indexing_mode_ = mode;
	}

	static void ScatterMin(const Tensor& tensor, const Tensor& value,
	                       const Tensors& indices, IndexingMode mode = IndexingMode::Clamp) {
		Tensor& a = MemoryOp("InterlockedMin", &tensor, indices, &value);
		a.node_->indexing_mode_ = mode;
	}

	static void ScatterOr(const Tensor& tensor, const Tensor& value,
	                      const Tensors& indices, IndexingMode mode = IndexingMode::Clamp) {
		Tensor& a = MemoryOp("InterlockedOr", &tensor, indices, &value);
		a.node_->indexing_mode_ = mode;
	}

	static void ScatterAnd(const Tensor& tensor, const Tensor& value,
	                       const Tensors& indices, IndexingMode mode = IndexingMode::Clamp) {
		Tensor& a = MemoryOp("InterlockedAnd", &tensor, indices, &value);
		a.node_->indexing_mode_ = mode;
	}

	static void ScatterXor(const Tensor& tensor, const Tensor& value,
	                       const Tensors& indices, IndexingMode mode = IndexingMode::Clamp) {
		Tensor& a = MemoryOp("InterlockedXor", &tensor, indices, &value);
		a.node_->indexing_mode_ = mode;
	}

	static int GetAxis(int dims, int axis) {
		if (axis < 0) {
			axis = dims + axis;
		}
		return axis;
	}

	static Tensor& ReductionOP(string name, const Tensor& tensor, int axis = 0, bool keepdims = false);
	static Tensor& ScanOP(string name, const Tensor& tensor, int axis = 0);

	static Tensor& Sum(const Tensor& tensor, int axis = 0) {
		return ReductionOP("dim_sum", tensor, axis);
	}

	static Tensor& Norm(const Tensor& tensor, int axis = 0) {
		return ReductionOP("dim_norm", tensor, axis);
	}

	static Tensor& Mean(const Tensor& tensor, int axis = 0) {
		return ReductionOP("dim_mean", tensor, axis);
	}

	static Tensor& Max(const Tensor& tensor, int axis = 0) {
		return ReductionOP("dim_max", tensor, axis);
	}

	static Tensor& Any(const Tensor& tensor, int axis = 0) {
		return ReductionOP("dim_any", tensor, axis);
	}

	static Tensor& All(const Tensor& tensor, int axis = 0) {
		return ReductionOP("dim_all", tensor, axis);
	}

	static Tensor& Min(const Tensor& tensor, int axis = 0) {
		return ReductionOP("dim_min", tensor, axis);
	}

	static Tensor& PrefixSum(const Tensor& tensor, int axis = 0) {
		return ScanOP("dim_prefix_sum", tensor, axis);
	}

	static Tensor& Reverse(const Tensor& tensor, int axis = 0) {
		Tensors shape = tensor.GetShape();
		int dims = (int)shape.size();
		axis = GetAxis(dims, axis);
		Tensor& output = OpShape("dim_reverse", shape, &tensor);
		output.SetData(axis);
		return output;
	}

	static Tensor& Repeat(const Tensor& tensor, const Tensor& repeats, int axis = 0) {
		//check if repeats is a scalar
		if (repeats.GetDimension() != 0) {
			throw std::runtime_error("Repeats argument must be a scalar");
		}
		int dims = (int)tensor.GetDimension();
		axis = GetAxis(dims, axis);
		Tensors shape = tensor.GetShape();
		Tensors new_shape = Tensors();
		for (int i = 0; i < dims; i++) {
			if (i == axis) {
				new_shape.push_back(&(*shape[i] * repeats));
			} else {
				new_shape.push_back(shape[i]);
			}
		}
		Tensor& output = OpShape("dim_repeat", new_shape, &tensor);
		output.SetData(axis);
		return output;
	}

	static Tensor& SplitDim(const Tensor& tensor, int split_size = 128, int axis = 0) {
		ShapeInfo shapeinfo = tensor.GetShapeInfo();
		int dims = shapeinfo.dim;
		Tensors shape = shapeinfo.GetTensors();
		axis = GetAxis(dims, axis);
		Tensors new_shape = Tensors();
		for (int i = 0; i < dims; i++) {
			if (i == axis) {
				new_shape.push_back(&Tensor::Constant(split_size));
				new_shape.push_back(&((*shape[i] + Tensor::Constant(split_size - 1)) / Tensor::Constant(split_size)));
			} else {
				new_shape.push_back(shape[i]);
			}
		}
		Tensor& output = OpShape("dim_split", new_shape, &tensor);
		output.SetData({(uint)axis, (uint)split_size});
		return output;
	}

	static Tensor& MergeDim(const Tensor& tensor, int axis = 0, const Tensor* target_size = nullptr) {
		ShapeInfo shapeinfo = tensor.GetShapeInfo();
		int dims = shapeinfo.dim;
		Tensors shape = shapeinfo.GetTensors();
		axis = GetAxis(dims, axis);
		if(axis == 0) axis = 1;
		const Tensor* target_size_tensor = nullptr;
		if(target_size == nullptr) {
			target_size_tensor = &(*shape[axis] * *shape[axis+1]);
		} else {
			target_size_tensor = target_size;
		}
		axis = GetAxis(dims, axis);
		Tensors new_shape = Tensors();
		for (int i = 0; i < dims; i++) {
			if(i == axis) {
				new_shape.push_back(target_size_tensor);
			} else if(i != axis+1) {
				new_shape.push_back(shape[i]);
			}
		}
		Tensor& output = OpShape("dim_merge", new_shape, &tensor);
		output.SetData(axis);
		return output;
	}

	static Tensor& Transpose(const Tensor& tensor, const int axis1 = 1, const int axis2 = 0) {
		ShapeInfo shapeinfo = tensor.GetShapeInfo();

		int dims = std::max(std::max(axis1+1, axis2+1), std::max(shapeinfo.dim, -std::min(axis1, axis2)));
		int a1 = GetAxis(dims, axis1);
		int a2 = GetAxis(dims, axis2);
		shapeinfo.ExpandDimensionsTo(dims);
		Tensors shape = shapeinfo.GetTensors();
		//swap the axes
		std::swap(shape[a1], shape[a2]);
		Tensor& output = OpShape("transpose", shape, &tensor);
		//add data
		output.SetData({AsUint(a1), AsUint(a2)});
		return output;
	}

	//dot product of
	static Tensor& Dot(const Tensor& tensor1, const Tensor& tensor2, int axis = 0) {
		Tensors shape = tensor1.GetShape();
		int dims = (int)shape.size();
		axis = GetAxis(dims, axis);
		shape.erase(shape.begin() + axis);
		Tensor& output = OpShape("dot", shape, &tensor1, &tensor2);
		output.SetData(axis);
		return output;
	}

	static Tensor& Unsqueeze(const Tensor& tensor, int axis = 0) {
		Tensors shape = tensor.GetShape();
		int dims = (int)shape.size();
		if(axis < 0) {
			axis = dims + axis + 1;
		}
		axis = std::max(0, std::min(dims, axis));
		shape.insert(shape.begin() + axis, &Constant(1));
		Tensor& output = OpShape("unsqueeze", shape, &tensor);
		output.SetData(axis);
		return output;
	}

	static bool AreTensorsEqual(const Tensor& a, const Tensor& b);

	static Tensor& Squeeze(const Tensor& tensor, int axis = 0) {
		Tensors shape = tensor.GetShape();
		int dims = (int)shape.size();
		axis = GetAxis(dims, axis);
		if (shape[axis]->TryGetConstant() != 1) {
			throw std::runtime_error("Cannot squeeze a dimension that is not 1");
		}
		shape.erase(shape.begin() + axis);
		Tensor& output = OpShape("squeeze", shape, &tensor);
		output.SetData(axis);
		return output;
	}

	//takes two tensors [T1, T2, ..., Tn, M, N] and [Tm, .., Tn, N, K] and returns [T1, T2, ..., Tm, M, K]
	static Tensor& Matmul(const Tensor& a, const Tensor& b) {
		ShapeInfo shape_a = a.GetShapeInfo();
		ShapeInfo shape_b = b.GetShapeInfo();

		if (shape_a.dim < 2 && shape_b.dim < 2) {
			throw std::runtime_error("Matrix multiplication requires at least one 2D tensor");
		}

		if(shape_a.dim < 2) {
			shape_a.ExpandDimensionsTo(2);
		}
		if(shape_b.dim < 2) {
			shape_b.ExpandDimensionsTo(2);
		}

		Tensors shape_a_tensors = shape_a.GetTensors();
		Tensors shape_b_tensors = shape_b.GetTensors();

		//get shape of the result
		Tensors shape_c = Tensors();
		int dim_a = shape_a.dim;
		int dim_b = shape_b.dim;
		int max_dim = 0;
		Tensors max_shape = Tensors();
		//get the shape with most dimensions
		if (dim_a < dim_b) {
			max_dim = dim_b;
			max_shape = shape_b_tensors;
		} else {
			max_dim = dim_a;
			max_shape = shape_a_tensors;
		}

		shape_c.push_back(shape_b_tensors[0]);
		shape_c.push_back(shape_a_tensors[1]);
		for (int i = 2; i < max_dim; i++) {
			shape_c.push_back(max_shape[i]);
		}
		ShapeDimCompareResult result = CompareShapeDim(shape_a_tensors[0]->node_, shape_b_tensors[1]->node_);
		if (!result.compatible) {
			throw std::runtime_error("Inner dimensions of the matrices must match");
		}

		Tensor& output = OpShape("matmul", shape_c, &a, &b);
		return output;
	}

	static Tensor& Reshape(const Tensor &tensor, const Tensors &shape, TFDataFormat format = TFTypeNone);
	static Tensor& Assert(const Tensor& tensor, const Tensors& shape, TFDataFormat type = TFTypeFloat32);

	Tensors enter_tensors = Tensors();
	bool already_entered = false;

	std::variant<Tensor*, Tensors> Enter()
	{
		if(!node_->op->HasAllTypes(OpProp::HasChildren)) {
			throw std::runtime_error("The node of type " + node_->name + " cannot have children");
		}

		if(already_entered) {
			throw std::runtime_error("Already entered node scope");
		}
		evaluation_context_ir_->BeginScopeLastChild(node_); //begin at the last child
		already_entered = true;
		if(enter_tensors.size() > 0) {
			return enter_tensors; //if we have some special info, like indices of kernel threads
		} else {
			return const_cast<Tensor*>(this);
		}
	}

	void Exit()
	{
		evaluation_context_ir_->EndScope();
	}

	static Tensor& Loop(const Tensor& start, const Tensor& end, const Tensor& step)
	{
		return Op("loop", &start, &end, &step);
	}

	static void Loop(const Tensor& start, const Tensor& end, const Tensor& step,
	                 const function<void(const Tensor&)>& body) {
		// create the loop
		Tensor& loop = Loop(start, end, step);

		evaluation_context_ir_->ExecuteExpressionFirstChild(loop.node_, [&]() {
			// create the body
			body(loop);
		});
	}

	static Tensor& If(const Tensor& condition) {
		// create the if
		Tensor& if_tensor = Op("if", &condition);
		return if_tensor;
	}

	static void If(const Tensor& condition,
		const std::function<void()>& body) {
		// create the if
		Tensor& if_tensor = If(condition);

		evaluation_context_ir_->ExecuteExpressionFirstChild(if_tensor.node_, [&]() {
			// create the body
			body();
		});
	}

	static void If(const Tensor& condition, const std::function<void()>& true_body,
		const std::function<void()>& false_body) {
		If(condition, true_body);
		If(!condition, false_body);
	}

	static void Vmap(const Tensors inputs, const Tensors shape, const std::function<void()>& body) {
		// create the if
		Tensor& vmap_main = OpShape("vmap", Tensors(), inputs);

		evaluation_context_ir_->ExecuteExpressionFirstChild(vmap_main.node_, [&]() {
			// create the body
			body();
		});
	}

	static Tensor& Kernel(const Tensors shape, vector<int> group_size = {}) {
		// create the kernel
		Tensor& kernel = Static("kernel", shape, TFTypeNone);
		evaluation_context_ir_->ExecuteExpressionFirstChild(kernel.node_, [&]() {
			for (int i = 0; i < shape.size(); i++) {
				kernel.enter_tensors.push_back(&Index(shape, i)); //thread indices
			}
		});
		kernel.node_->group_size = group_size;
		return kernel;
	}

	static Tensor& Kernel(const Tensors shape, const std::function<void(Tensors)>& body, vector<int> group_size = {}) {
		// create the kernel
		Tensor& kernel = Kernel(shape);

		evaluation_context_ir_->ExecuteExpressionLastChild(kernel.node_, [&]() {
			// create the body
			body(kernel.enter_tensors);
		});

		kernel.node_->group_size = group_size;
		return kernel;
	}

	static Tensor& Kernel(const NodeArguments& shape)
	{
		// create the kernel
		Tensor& kernel = Static("kernel", shape, TFTypeNone);
		return kernel;
	}

	static void Break() {
		// create the break
		Tensor& break_tensor = Static("break", TFTypeNone);
	}

	static void Continue() {
		// create the continue
		Tensor& continue_tensor = Static("continue", TFTypeNone);
	}

	// destructor
	~Tensor() = default;

	Tensor& operator-() const { return Op("neg", this); }
	Tensor& operator!() const {
		if(node_->format.type == TFType::Bool) {
			return Op("notb", this);
		} else {
			return Op("not", this);
		}
	}
	Tensor& operator~() const {
		if(node_->format.type == TFType::Bool) {
			return Op("notb", this);
		} else {
			return Op("not", this);
		}
	}

	Tensor& operator+(const Tensor& other) const {
		return Op("add", this, &other);
	}

	Tensor& operator-(const Tensor& other) const {
		return Op("sub", this, &other);
	}

	Tensor& operator*(const Tensor& other) const {
		return Op("mul", this, &other);
	}

	Tensor& operator/(const Tensor& other) const {
		return Op("div", this, &other);
	}

	Tensor& operator%(const Tensor& other) const {
		return Op("mod", this, &other);
	}

	Tensor& operator>(const Tensor& other) const {
		return Op("gt", this, &other);
	}

	Tensor& operator<(const Tensor& other) const {
		return Op("lt", this, &other);
	}

	Tensor& operator>=(const Tensor& other) const {
		return Op("gte", this, &other);
	}

	Tensor& operator<=(const Tensor& other) const {
		return Op("lte", this, &other);
	}

	Tensor& operator==(const Tensor& other) const {
		return Op("eq", this, &other);
	}

	Tensor& operator!=(const Tensor& other) const {
		return Op("neq", this, &other);
	}

	Tensor& operator&&(const Tensor& other) const {
		return Op("and", this, &other);
	}

	Tensor& operator||(const Tensor& other) const {
		return Op("or", this, &other);
	}

	Tensor& operator&(const Tensor& other) const {
		return Op("and", this, &other);
	}

	Tensor& operator|(const Tensor& other) const {
		return Op("or", this, &other);
	}

	Tensor& operator^(const Tensor& other) const {
		return Op("xor", this, &other);
	}

	Tensor& operator<<(const Tensor& other) const {
		return Op("lshift", this, &other);
	}

	Tensor& operator>>(const Tensor& other) const {
		return Op("rshift", this, &other);
	}

	void operator=(const Tensor& other) = delete;

	static Tensor& copy(const Tensor& tensor) {
		return Op("copy", &tensor);
	}

	static Tensor& sin(const Tensor& x) { return Op("sin", &x); }
	static Tensor& cos(const Tensor& x) { return Op("cos", &x); }
	static Tensor& tan(const Tensor& x) { return Op("tan", &x); }
	static Tensor& asin(const Tensor& x) { return Op("asin", &x); }
	static Tensor& acos(const Tensor& x) { return Op("acos", &x); }
	static Tensor& atan(const Tensor& x) { return Op("atan", &x); }
	static Tensor& sinh(const Tensor& x) { return Op("sinh", &x); }
	static Tensor& cosh(const Tensor& x) { return Op("cosh", &x); }
	static Tensor& tanh(const Tensor& x) { return Op("tanh", &x); }
	static Tensor& asinh(const Tensor& x) { return Op("asinh", &x); }
	static Tensor& acosh(const Tensor& x) { return Op("acosh", &x); }
	static Tensor& atanh(const Tensor& x) { return Op("atanh", &x); }
	static Tensor& exp(const Tensor& x) { return Op("exp", &x); }
	static Tensor& log(const Tensor& x) { return Op("log", &x); }
	static Tensor& log2(const Tensor& x) { return Op("log2", &x); }
	static Tensor& exp2(const Tensor& x) { return Op("exp2", &x); }
	static Tensor& sqrt(const Tensor& x) { return Op("sqrt", &x); }
	static Tensor& sqr(const Tensor& x) { return Op("sqr", &x); }
	static Tensor& rsqrt(const Tensor& x) { return Op("rsqrt", &x); }
	static Tensor& rcp(const Tensor& x) { return Op("rcp", &x); }
	static Tensor& abs(const Tensor& x) { return Op("abs", &x); }
	static Tensor& sign(const Tensor& x) { return Op("sign", &x); }
	static Tensor& floor(const Tensor& x) { return Op("floor", &x); }
	static Tensor& ceil(const Tensor& x) { return Op("ceil", &x); }
	static Tensor& round(const Tensor& x) { return Op("round", &x); }
	static Tensor& trunc(const Tensor& x) { return Op("trunc", &x); }
	static Tensor& frac(const Tensor& x) { return Op("frac", &x); }

	static Tensor& pcg(const Tensor& x) { return Op("pcg", &x); }
	static Tensor& pcgf(const Tensor& x) { return Op("pcgf", &x); }

	static Tensor& reversebits(const Tensor& x) { return Op("reversebits", &x); }

	static Tensor& tofloat(const Tensor& x) { return Op("float", &x); }
	static Tensor& toint(const Tensor& x) { return Op("int", &x); }
	static Tensor& touint(const Tensor& x) { return Op("uint", &x); }
	static Tensor& tobool(const Tensor& x) { return Op("bool", &x); }

	static Tensor& asfloat(const Tensor& x) { return Op("asfloat", &x); }
	static Tensor& asint(const Tensor& x) { return Op("asint", &x); }
	static Tensor& asuint(const Tensor& x) { return Op("asuint", &x); }

	static Tensor& clamp(const Tensor& x, const Tensor& min, const Tensor& max) {
		return Op("clamp", &x, &min, &max);
	}

	static Tensor& pow(const Tensor& x, const Tensor& y) {
		return Op("pow", &x, &y);
	}

	static Tensor& min(const Tensor& x, const Tensor& y) {
		return Op("min", &x, &y);
	}

	static Tensor& max(const Tensor& x, const Tensor& y) {
		return Op("max", &x, &y);
	}

	static Tensor& mod(const Tensor& x, const Tensor& y) {
		return Op("mod", &x, &y);
	}

	static Tensor& modf(const Tensor& x, const Tensor& y) {
		return Op("modf", &x, &y);
	}

	static Tensor& atan2(const Tensor& x, const Tensor& y) {
		return Op("atan2", &x, &y);
	}

	static Tensor& grad(const Tensor& x, const Tensor& wrt) {
		if(x.node_->op->HasAllTypes(OpProp::Nondiff) && !x.node_->flags.has(NodeProp::Modified)) {
			throw std::runtime_error("Cannot compute gradient of a non-differentiable operation");
		}
		return OpShape("backwards_grad", wrt.GetShape(), &x, &wrt);
	}

	static Tensor& lerp(const Tensor& x, const Tensor& y, const Tensor& a) {
		return Op("lerp", &x, &y, &a);
	}

	static Tensor& smoothstep(const Tensor& a, const Tensor& b, const Tensor& x) {
		return Op("smoothstep", &a, &b, &x);
	}

	static Tensor& select(const Tensor& cond, const Tensor& x, const Tensor& y) {
		return Op("ternary", &cond, &x, &y);
	}

	static Tensor& fma(const Tensor& x, const Tensor& y, const Tensor& z) {
		return Op("fma", &x, &y, &z);
	}

	static Tensors IndexGrid(const Tensors& begin, const Tensors& end) {
		//compute shape	
		Tensors shape = Tensors();
		for (int i = 0; i < begin.size(); i++) {
			shape.push_back(&(*end[i] - *begin[i]));
		}
		//compute indices
		Tensors index_grid = Tensors();
		for (int i = 0; i < begin.size(); i++) {
			index_grid.push_back(&(Index(shape, i) + *begin[i]));
		}
		return index_grid;
	}

	static Tensors IndexGrid(const Tensors& begin, const Tensors& end, const Tensors& step)
	{
		Tensors shape = Tensors();
		for (int i = 0; i < begin.size(); i++) {
			shape.push_back(&((*end[i] - *begin[i]) / *step[i]));
		}
		//compute indices
		Tensors index_grid = Tensors();
		for (int i = 0; i < begin.size(); i++) {
			index_grid.push_back(&(Index(shape, i) * *step[i] + *begin[i]));
		}
		return index_grid;
	}

	static void PrintValue(std::string name, const Tensor& tensor) {
		if (tensor.GetDimension() != 0) {
			throw std::runtime_error("Cannot print a non-scalar value");
		}
		Tensor& output = Op("print_value", &tensor);
		output.SetDebugName(name);
	}

	static void AssertValue(std::string name, const Tensor& tensor) {
		if (tensor.GetDimension() != 0) {
			throw std::runtime_error("Cannot assert a non-scalar value");
		}
		Tensor& output = Op("assert_value", &tensor);
		output.SetDebugName(name);
	}

	void SetDebugName(const string& name) const;

	static void BeginRegion(const string& name);
	static void EndRegion(const string& name);

	int axis(int i = 0) const {
		return (int)node_->data[i];
	}

	uint data(int i = 0) const {
		return node_->data[i];
	}
};

}  // namespace TensorFrost
