#pragma once

#include <list>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include <math.h>

#include "IR/IR.h"
#include "Utility/Utility.h"

namespace TensorFrost {

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
	const Tensor* tensor;
	int index;

	Argument(Type type, const Tensor* tensor, int index)
	    : type(type), tensor(tensor), index(index) {}
};

using Tensors = vector<const Tensor*>;
using Arguments = vector<Argument>;

class Tensor {
 private:
	static void AddArguments(Arguments& arguments, const Tensors& tensors,
	                         Argument::Type type) {
		for (int i = 0; i < tensors.size(); i++) {
			arguments.emplace_back(type, tensors[i], i);
		}
	}

	static void AddArguments(Arguments& arguments, const Arguments& toadd) {
		for (const auto& i : toadd) {
			arguments.push_back(i);
		}
	}

	template <typename... Args>
	static shared_ptr<Tensor> Op(const std::string& op, const Args*... args) {
		// convert the parameter pack to a std::vector
		Tensors tensors = {args...};

		// create argument list
		Arguments arguments = Arguments();

		AddArguments(arguments, tensors, Argument::Type::Input);
		AddArguments(arguments, tensors[0]->GetArguments(Argument::Type::Shape));

		shared_ptr<Tensor> output =
		    make_shared<Tensor>(arguments, op, tensors[0]->type);

		// create the output tensor
		AddToGraph(output);
		return output;
	}

	template <typename... Args>
	static shared_ptr<Tensor> IndexedOp(const string op, const Tensors indices,
	                                    const Args*... args) {
		// convert the parameter pack to a std::vector
		Tensors tensors = {args...};

		// create argument list
		Arguments arguments = Arguments();

		// add the input tensors
		AddArguments(arguments, tensors, Argument::Type::Input);

		// add the indices
		AddArguments(arguments, indices, Argument::Type::Index);

		// add the shape
		AddArguments(arguments, indices[0]->GetArguments(Argument::Type::Shape));

		// create the output tensor
		shared_ptr<Tensor> output =
		    make_shared<Tensor>(arguments, op, tensors[0]->type);

		// create the output tensor
		AddToGraph(output);

		return output;
	}

	static shared_ptr<Tensor> Static(const string& op) {
		shared_ptr<Tensor> output =
		    make_shared<Tensor>(Arguments(), op, DataType::Float);
		AddToGraph(output);
		return output;
	}
	static shared_ptr<Tensor> Static(const string& op, const Tensors& shape) {
		Arguments arguments = Arguments();
		AddArguments(arguments, shape, Argument::Type::Shape);
		shared_ptr<Tensor> output =
		    make_shared<Tensor>(arguments, op, DataType::Float);
		AddToGraph(output);
		return output;
	}
	static shared_ptr<Tensor> Static(const string& op, const Arguments& shape) {
		Arguments arguments = Arguments();
		AddArguments(arguments, shape);
		shared_ptr<Tensor> output =
		    make_shared<Tensor>(arguments, op, DataType::Float);
		AddToGraph(output);
		return output;
	}

	static IR* evaluation_context_ir_;

	static void AddToGraph(const shared_ptr<Tensor>& node) {
		// check if IR is not null
		if (evaluation_context_ir_ == nullptr) {
			throw std::runtime_error("Evaluation context has not been set.");
		}

		evaluation_context_ir_->AddNode(node);
	}

 public:
	static void SetEvaluationContext(IR* ir) {
		 if(evaluation_context_ir_ != nullptr && ir != nullptr) {
			 throw std::runtime_error("Evaluation context change is forbidden.");
		 }
		 evaluation_context_ir_ = ir; 
	}

	[[nodiscard]] string GetConstantString() const;

	string name;
	DataType type = DataType::Float;
	Arguments inputs;
	std::vector<uint> data;

	// Main constructor
	Tensor(Arguments inputs, string name, DataType type) {
		this->inputs = std::move(inputs);
		this->name = std::move(name);
		this->type = type;
	}

	[[nodiscard]] Arguments GetArguments(Argument::Type type) const {
		Arguments result = Arguments();
		for (const auto& input : inputs) {
			if (input.type == type) {
				result.push_back(input);
			}
		}
		return result;
	}
	[[nodiscard]] Tensors GetArgumentTensors(Argument::Type type) const {
		Tensors result = Tensors();
		for (const auto& input : inputs) {
			if (input.type == type) {
				result.push_back(input.tensor);
			}
		}
		return result;
	}
	[[nodiscard]] vector<const Tensor*> GetShape() const {
		vector<const Tensor*> result = vector<const Tensor*>();
		// get max dimension
		int max_dim = -1;
		for (const auto& input : inputs) {
			if (input.type == Argument::Type::Shape) {
				max_dim = std::max(max_dim, input.index);
			}
		}

		if (max_dim == -1) {
			return result;
		}

		// resize result
		result.resize(max_dim + 1);
		for (int i = 0; i <= max_dim; i++) {
			result[i] = nullptr;
		}
		// fill result
		for (const auto& input : inputs) {
			if (input.type == Argument::Type::Shape) {
				result[input.index] = input.tensor;
			}
		}
		// if there are any missing dimensions, fill them with 1
		Tensor& one = Constant(1);
		for (int i = 0; i <= max_dim; i++) {
			if (result[i] == nullptr) {
				result[i] = &one;
			}
		}
		return result;
	}
	[[nodiscard]] vector<int> TryGetShape() const {
		vector<int> result = vector<int>();
		// get max dimension
		int max_dim = -1;
		for (const auto& input : inputs) {
			if (input.type == Argument::Type::Shape) {
				max_dim = std::max(max_dim, input.index);
			}
		}

		if (max_dim == -1) {
			return result;
		}

		// resize result
		result.resize(max_dim + 1);
		for (int i = 0; i <= max_dim; i++) {
			result[i] = 1;
		}
		// fill result
		for (const auto& input : inputs) {
			if (input.type == Argument::Type::Shape) {
				result[input.index] = AsInt(input.tensor->data[0]);
			}
		}
		return result;
	}

	// tensor factory methods
	static Tensors GetConstantShape(const vector<int>& shape) {
		Tensors result = vector<const Tensor*>();
		for (int i : shape) {
			result.push_back(&Constant(i));
		}
		return result;
	}

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

	static Tensor& Constant(const vector<int>& shape, float* data) {
		shared_ptr<Tensor> output = Static("const_memory", GetConstantShape(shape));
		int data_count = GetSize(shape);
		for (int i = 0; i < data_count; i++) {
			output->data.push_back(AsUint(data[i]));
		}
		output->type = DataType::Float;
		return *output;
	}
	static Tensor& Constant(const Tensors& shape, float value) {
		Tensor& output = Constant(value);
		AddArguments(output.inputs, shape, Argument::Type::Shape);
		return output;
	}
	static Tensor& Constant(const vector<int>& shape, float value) {
		return Constant(GetConstantShape(shape), value);
	}
	static Tensor& Constant(const Tensors& shape, int value) {
		Tensor& output = Constant(value);
		AddArguments(output.inputs, shape, Argument::Type::Shape);
		return output;
	}
	static Tensor& Constant(const vector<int>& shape, int value) {
		return Constant(GetConstantShape(shape), value);
	}
	static Tensor& Constant(const Tensors& shape, uint value) {
		Tensor& output = Constant(value);
		AddArguments(output.inputs, shape, Argument::Type::Shape);
		return output;
	}
	static Tensor& Constant(const vector<int>& shape, uint value) {
		return Constant(GetConstantShape(shape), value);
	}
	static Tensor& Input() {
		shared_ptr<Tensor> output = Static("input_memory");
		output->type = DataType::Float;
		return *output;
	}
	static Tensor& Input(const Tensors& shape) {
		shared_ptr<Tensor> output = Static("input_memory", shape);
		output->type = DataType::MemoryRef;
		return *output;
	}
	static vector<const Tensor*> GetInputShape(const vector<int>& shape) {
		Tensors result = vector<const Tensor*>();
		for (int i : shape) {
			if (i < 0) {
				result.push_back(&Input());
			} else {
				result.push_back(&Constant(i));
			}
		}
		return result;
	}
	static Tensor& Input(const vector<int>& shape) {
		return Input(GetInputShape(shape));
	}

	static Tensor& Index(const Tensors& shape, int dim) {
		shared_ptr<Tensor> output = Static("dim_id", shape);
		output->data = std::vector<uint>(1, dim);
		output->type = DataType::Int;
		return *output;
	}
	static Tensor& Load(const Tensor& tensor, const Tensors& indices) {
		return *IndexedOp("load", indices, &tensor);
	}

	[[nodiscard]] Tensor& Index(int dim) const {
		shared_ptr<Tensor> output =
		    Static("dim_id", this->GetArguments(Argument::Type::Shape));
		output->data = std::vector<uint>(1, dim);
		output->type = DataType::Int;
		return *output;
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
