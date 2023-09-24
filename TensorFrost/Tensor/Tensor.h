#pragma once

#include <list>
#include <unordered_map>
#include <utility>
#include <vector>
#include <math.h>
#include "IR/IR.h"
#include "Utility/Utility.h"

namespace TensorFrost {

typedef std::vector<const Tensor*> Tensors;

class Tensor;

class Argument {
 public:
	enum Type {
		Input,
		Index,
		Shape,
		RefCopy,
		Loop,
	};

	Type type;
	Tensor* tensor;
	int index;

	Argument(Type type, Tensor* tensor, int index) {
		this->type = type;
		this->tensor = tensor;
		this->index = index;
	}
};

class Tensor {
 private:
	template <typename... Args>
	static shared_ptr<Tensor> Op(const std::string& op, const Args*... args) {
		// convert the parameter pack to a std::vector
		Tensors tensors = {args...};

		// create argument list
		std::vector<Argument> arguments = std::vector<Argument>();

		for (int i = 0; i < tensors.size(); i++) {
			arguments.emplace_back(Argument::Type::Input,
			                       const_cast<Tensor*>(tensors[i]), i);
		}

		shared_ptr<Tensor> output =
		    make_shared<Tensor>(arguments, op);

		// create the output tensor
		AddToGraph(output);
		return output;
	}

	static void AddArguments(std::vector<Argument>& arguments,
	                         const Tensors& tensors, Argument::Type type) {
		for (int i = 0; i < tensors.size(); i++) {
			arguments.emplace_back(type, const_cast<Tensor*>(tensors[i]), i);
		}
	}

	template <typename... Args>
	static shared_ptr<Tensor> IndexedOp(const string op, const Tensors indices,
	                                    const Args*... args) {
		// convert the parameter pack to a std::vector
		Tensors tensors = {args...};

		// create argument list
		std::vector<Argument> arguments = std::vector<Argument>();

		// add the input tensors
		AddArguments(arguments, tensors, Argument::Type::Input);

		// add the indices
		AddArguments(arguments, indices, Argument::Type::Index);

		// create the output tensor
		shared_ptr<Tensor> output =
		    make_shared<Tensor>(arguments, op);

		// create the output tensor
		AddToGraph(output);

		return output;
	}

	static shared_ptr<Tensor> Static(const string& op) {
		shared_ptr<Tensor> output =
		    make_shared<Tensor>(vector<Argument>(), op);
		AddToGraph(output);
		return output;
	}

	static shared_ptr<Tensor> Static(const string& op, const vector<const Tensor*>& shape) {
		vector<Argument> arguments = vector<Argument>();
		AddArguments(arguments, shape, Argument::Type::Shape);
		shared_ptr<Tensor> output =
		    make_shared<Tensor>(arguments, op);
		AddToGraph(output);
		return output;
	}

	static IR* graph_;

	static void AddToGraph(const shared_ptr<Tensor>& node) {
		// check if IR is not null
		if (graph_ == nullptr) {
			throw std::runtime_error(
			    "Operation cannot be added to graph because the graph is null");
		}

		graph_->AddNode(node);
	}

 public:
	static void SetIR(IR* ir) { graph_ = ir; }

	string GetConstantString() const {
		if (name == "const" || name == "dim_id") {
			switch (type) {
				case DataType::Float:
					return to_string(AsFloat(data[0]));
				case DataType::Int:
					return to_string(AsInt(data[0]));
				case DataType::Uint:
					return to_string(data[0]);
				default:
					return "";
			}
		} else {
			return "";
		}
	}

	string name;
	DataType type = DataType::Float;
	std::vector<Argument> inputs;
	std::vector<uint> data;

	// Main constructor
	Tensor(vector<Argument> inputs, string name = "",
	       DataType type = DataType::Float) {
		this->inputs = std::move(inputs);
		this->name = std::move(name);
		this->type = type;
	}

	vector<const Tensor*> GetShape() const
	{
		vector<const Tensor*> result = vector<const Tensor*>();
		//get max dimension
		int max_dim = -1;
		for (int i = 0; i < inputs.size(); i++) {
			if (inputs[i].type == Argument::Type::Shape) {
				max_dim = std::max(max_dim, inputs[i].index);
			}
		}

		if(max_dim == -1)
		{
			return result;
		}

		//resize result
		result.resize(max_dim + 1);
		for(int i = 0; i <= max_dim; i++)
		{
			result[i] = nullptr;
		}
		//fill result
		for (int i = 0; i < inputs.size(); i++) {
			if (inputs[i].type == Argument::Type::Shape) {
				result[inputs[i].index] = inputs[i].tensor;
			}
		}
		//if there are any missing dimensions, fill them with 1
		Tensor& one = Constant(1);
		for(int i = 0; i <= max_dim; i++)
		{
			if(result[i] == nullptr)
			{
				result[i] = &one;
			}
		}
		return result;
	}

	vector<int> TryGetShape() const
	{
		vector<int> result = vector<int>();
		//get max dimension
		int max_dim = -1;
		for (int i = 0; i < inputs.size(); i++) {
			if (inputs[i].type == Argument::Type::Shape) {
				max_dim = std::max(max_dim, inputs[i].index);
			}
		}

		if(max_dim == -1)
		{
			return result;
		}

		//resize result
		result.resize(max_dim + 1);
		for(int i = 0; i <= max_dim; i++)
		{
			result[i] = 1;
		}
		//fill result
		for (int i = 0; i < inputs.size(); i++) {
			if (inputs[i].type == Argument::Type::Shape) {
				result[inputs[i].index] = AsInt(inputs[i].tensor->data[0]);
			}
		}
		return result;
	}

	// tensor factory methods
	static Tensor& Constant(float value) {
		shared_ptr<Tensor> output = Static("const");
		output->data = std::vector<uint>(1, AsUint(value));
		output->type = DataType::Float;
		return *output;
	}

	static Tensor& Constant(int value) {
		shared_ptr<Tensor> output = Static("const");
		output->data = std::vector<uint>(1, AsUint(value));
		output->type = DataType::Int;
		return *output;
	}

	static Tensor& Constant(uint value) {
		shared_ptr<Tensor> output = Static("const");
		output->data = std::vector<uint>(1, value);
		output->type = DataType::Uint;
		return *output;
	}

	static vector<const Tensor*> GetConstantShape(vector<int> shape) {
		vector<const Tensor*> result = vector<const Tensor*>();
		for (int i = 0; i < shape.size(); i++) {
			result.push_back(&Constant(shape[i]));
		}
		return result;
	}

	static Tensor& Constant(const vector<int> shape, float* data) {
		shared_ptr<Tensor> output = Static("const_memory", GetConstantShape(shape));
		int data_count = GetSize(shape);
		for (int i = 0; i < data_count; i++) {
			output->data.push_back(AsUint(data[i]));
		}
		output->type = DataType::Float;
		return *output;
	}

	static Tensor& Constant(const Tensors& shape, float value)
	{
		Tensor& output = Constant(value);
		AddArguments(output.inputs, shape, Argument::Type::Shape);
		return output;
	}

	static Tensor& Constant(const vector<int> shape, float value)
	{
		return Constant(GetConstantShape(shape), value);
	}

	static Tensor& Constant(const Tensors& shape, int value)
	{
		Tensor& output = Constant(value);
		AddArguments(output.inputs, shape, Argument::Type::Shape);
		return output;
	}

	static Tensor& Constant(const vector<int> shape, int value)
	{
		return Constant(GetConstantShape(shape), value);
	}

	static Tensor& Constant(const Tensors& shape, uint value)
	{
		Tensor& output = Constant(value);
		AddArguments(output.inputs, shape, Argument::Type::Shape);
		return output;
	}

	static Tensor& Constant(const vector<int> shape, uint value)
	{
		return Constant(GetConstantShape(shape), value);
	}

	static Tensor& Input(const Tensors shape) {
		shared_ptr<Tensor> output = Static("input_memory", shape);
		output->type = DataType::MemoryRef;
		return *output;
	}

	static Tensor& Input(const vector<int> shape) {
		return Input(GetConstantShape(shape));
	}

	static Tensor& Index(const Tensors shape, int dim) {
		shared_ptr<Tensor> output = Static("dim_id", shape);
		output->data = std::vector<uint>(1, dim);
		output->type = DataType::Int;
		return *output;
	}

	static Tensor& Load(const Tensor& tensor, const Tensors& indices) {
		return *IndexedOp("load", indices, &tensor);
	}

	static void Store(const Tensor& tensor, const Tensor& value,
	                  const Tensors& indices) {
		IndexedOp("store", indices, &tensor, &value);
	}

	static void ScatterAdd(const Tensor& tensor, const Tensor& value,
	                       const Tensors& indices) {
		IndexedOp("InterlockedAdd", indices, &tensor, &value);
	}

	static void ScatterMax(const Tensor& tensor, const Tensor& value,
	                       const Tensors& indices) {
		IndexedOp("InterlockedMax", indices, &tensor, &value);
	}

	static void ScatterMin(const Tensor& tensor, const Tensor& value,
	                       const Tensors& indices) {
		IndexedOp("InterlockedMin", indices, &tensor, &value);
	}

	static void ScatterOr(const Tensor& tensor, const Tensor& value,
	                      const Tensors& indices) {
		IndexedOp("InterlockedOr", indices, &tensor, &value);
	}

	static void ScatterAnd(const Tensor& tensor, const Tensor& value,
	                       const Tensors& indices) {
		IndexedOp("InterlockedAnd", indices, &tensor, &value);
	}

	static void ScatterXor(const Tensor& tensor, const Tensor& value,
	                       const Tensors& indices) {
		IndexedOp("InterlockedXor", indices, &tensor, &value);
	}

	Tensor& Index(int dim) const { return Index(this->GetShape(), dim); }

	// destructor
	~Tensor() = default;

	Tensor& operator-() const { return *Op("neg", this); }
	Tensor& operator!() const { return *Op("not", this); }
	Tensor& operator~() const { return *Op("bnot", this); }

	Tensor& operator+(const Tensor& other) const {
		return *Op("add", this, &other);
	}

	Tensor& operator-(const Tensor& other) const {
		return *Op("sub", this, &other);
	}

	Tensor& operator*(const Tensor& other) const {
		return *Op("mul", this, &other);
	}

	Tensor& operator/(const Tensor& other) const {
		return *Op("div", this, &other);
	}

	Tensor& operator%(const Tensor& other) const {
		return *Op("mod", this, &other);
	}

	Tensor& operator>(const Tensor& other) const {
		return *Op("gt", this, &other);
	}

	Tensor& operator<(const Tensor& other) const {
		return *Op("lt", this, &other);
	}

	Tensor& operator>=(const Tensor& other) const {
		return *Op("gte", this, &other);
	}

	Tensor& operator<=(const Tensor& other) const {
		return *Op("lte", this, &other);
	}

	Tensor& operator==(const Tensor& other) const {
		return *Op("eq", this, &other);
	}

	Tensor& operator!=(const Tensor& other) const {
		return *Op("neq", this, &other);
	}

	Tensor& operator&&(const Tensor& other) const {
		return *Op("and", this, &other);
	}

	Tensor& operator||(const Tensor& other) const {
		return *Op("or", this, &other);
	}


	Tensor& operator&(const Tensor& other) const {
		return *Op("band", this, &other);
	}

	Tensor& operator|(const Tensor& other) const {
		return *Op("bor", this, &other);
	}

	Tensor& operator^(const Tensor& other) const {
		return *Op("bxor", this, &other);
	}

	Tensor& operator<<(const Tensor& other) const {
		return *Op("blshift", this, &other);
	}

	Tensor& operator>>(const Tensor& other) const {
		return *Op("brshift", this, &other);
	}

	static Tensor& ifcond(const Tensor& condition, const Tensor& ifTrue,
	                      const Tensor& ifFalse) {
		return *Op("cond", &condition, &ifTrue, &ifFalse);
	}

	static Tensor& sin(const Tensor& x) { return *Op("sin", &x); }
	static Tensor& cos(const Tensor& x) { return *Op("cos", &x); }
	static Tensor& tan(const Tensor& x) { return *Op("tan", &x); }
	static Tensor& asin(const Tensor& x) { return *Op("asin", &x); }
	static Tensor& acos(const Tensor& x) { return *Op("acos", &x); }
	static Tensor& atan(const Tensor& x) { return *Op("atan", &x); }
	static Tensor& sinh(const Tensor& x) { return *Op("sinh", &x); }
	static Tensor& cosh(const Tensor& x) { return *Op("cosh", &x); }
	static Tensor& tanh(const Tensor& x) { return *Op("tanh", &x); }
	static Tensor& asinh(const Tensor& x) { return *Op("asinh", &x); }
	static Tensor& acosh(const Tensor& x) { return *Op("acosh", &x); }
	static Tensor& atanh(const Tensor& x) { return *Op("atanh", &x); }
	static Tensor& exp(const Tensor& x) { return *Op("exp", &x); }
	static Tensor& log(const Tensor& x) { return *Op("log", &x); }
	static Tensor& log2(const Tensor& x) { return *Op("log2", &x); }
	static Tensor& exp2(const Tensor& x) { return *Op("exp2", &x); }
	static Tensor& sqrt(const Tensor& x) { return *Op("sqrt", &x); }
	static Tensor& sqr(const Tensor& x) { return *Op("sqr", &x); }
	static Tensor& rsqrt(const Tensor& x) { return *Op("rsqrt", &x); }
	static Tensor& rcp(const Tensor& x) { return *Op("rcp", &x); }
	static Tensor& abs(const Tensor& x) { return *Op("abs", &x); }
	static Tensor& sign(const Tensor& x) { return *Op("sign", &x); }
	static Tensor& floor(const Tensor& x) { return *Op("floor", &x); }
	static Tensor& ceil(const Tensor& x) { return *Op("ceil", &x); }
	static Tensor& round(const Tensor& x) { return *Op("round", &x); }
	static Tensor& trunc(const Tensor& x) { return *Op("trunc", &x); }
	static Tensor& frac(const Tensor& x) { return *Op("frac", &x); }

	static Tensor& clamp(const Tensor& x, const Tensor& min, const Tensor& max) {
		return *Op("clamp", &x, &min, &max);
	}

	static Tensor& pow(const Tensor& x, const Tensor& y) {
		return *Op("pow", &x, &y);
	}

	static Tensor& min(const Tensor& x, const Tensor& y) {
		return *Op("min", &x, &y);
	}

	static Tensor& max(const Tensor& x, const Tensor& y) {
		return *Op("max", &x, &y);
	}

	static Tensor& mod(const Tensor& x, const Tensor& y) {
		return *Op("mod", &x, &y);
	}

	static Tensor& atan2(const Tensor& x, const Tensor& y) {
		return *Op("atan2", &x, &y);
	}

	static Tensor& lerp(const Tensor& x, const Tensor& y, const Tensor& a) {
		return *Op("lerp", &x, &y, &a);
	}

	static Tensor& fma(const Tensor& x, const Tensor& y, const Tensor& z) {
		return *Op("fma", &x, &y, &z);
	}
};

}  // namespace TensorFrost
