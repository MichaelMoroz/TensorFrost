#pragma once

#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

#include "IR/IR.h"
#include "Tensor/Shape.h"
#include "Utility/Utility.h"

namespace TensorFrost {
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
		std::vector<const Tensor*> tensors = {args...};

		// create argument list
		std::vector<Argument> arguments = std::vector<Argument>();

		for (int i = 0; i < tensors.size(); i++) {
			arguments.emplace_back(Argument::Type::Input,
			                       const_cast<Tensor*>(tensors[i]), i);
		}

		shared_ptr<Tensor> output =
		    make_shared<Tensor>(tensors[0]->shape, arguments, op);

		// create the output tensor
		AddToGraph(output);
		return output;
	}

	template <typename... Args>
	static shared_ptr<Tensor> IndexedOp(const string op,
	                                    const vector<const Tensor*> indices,
	                                    const Args*... args) {
		// convert the parameter pack to a std::vector
		std::vector<const Tensor*> tensors = {args...};

		// create argument list
		std::vector<Argument> arguments = std::vector<Argument>();

		// add the input tensors
		for (int i = 0; i < tensors.size(); i++) {
			arguments.emplace_back(Argument::Type::Input,
			                       const_cast<Tensor*>(tensors[i]), i);
		}

		// add the indices
		for (int i = 0; i < indices.size(); i++) {
			arguments.emplace_back(Argument::Type::Index,
			                       const_cast<Tensor*>(indices[i]), i);
		}

		// create the output tensor
		shared_ptr<Tensor> output =
		    make_shared<Tensor>(tensors[0]->shape, arguments, op);

		// create the output tensor
		AddToGraph(output);

		return output;
	}

	static shared_ptr<Tensor> Static(const string& op, const Shape& shape) {
		shared_ptr<Tensor> output =
		    make_shared<Tensor>(shape, vector<Argument>(), op);
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

	string GetConstantString() {
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
	Shape shape;
	std::vector<uint> data;

	// Main constructor
	Tensor(Shape shape, vector<Argument> inputs, string name = "",
	       DataType type = DataType::Float) {
		this->shape = std::move(shape);
		this->inputs = std::move(inputs);
		this->name = std::move(name);
		this->type = type;
	}

	// Constructor that takes a list of sizes for each dimension
	explicit Tensor(std::vector<int> sizes, float value = 0.0)
	    : Tensor(Shape(sizes), vector<Argument>(), "const", DataType::Float) {
		data = std::vector<uint>(1, AsUint(value));
	}

	Tensor(std::vector<int> sizes, int value)
	    : Tensor(Shape(sizes), vector<Argument>(), "const", DataType::Int) {
		data = std::vector<uint>(1, AsUint(value));
	}

	Tensor(std::vector<int> sizes, uint value)
	    : Tensor(Shape(sizes), vector<Argument>(), "const", DataType::Uint) {
		data = std::vector<uint>(1, value);
	}

	explicit Tensor(float value) : Tensor({1}, value) {}

	explicit Tensor(int value) : Tensor({1}, value) {}

	explicit Tensor(uint value) : Tensor({1}, value) {}

	Tensor(float* data, std::vector<int> sizes)
	    : Tensor(Shape(sizes), vector<Argument>(), "const_memory",
	             DataType::Float) {
		uint data_count = Size();
		for (int i = 0; i < data_count; i++) {
			this->data.push_back(AsUint(data[i]));
		}
	}

	// tensor factory methods
	static Tensor& Constant(const Shape& shape, float value) {
		shared_ptr<Tensor> output = Static("const", shape);
		output->data = std::vector<uint>(1, AsUint(value));
		output->type = DataType::Float;
		return *output;
	}

	static Tensor& Constant(const Shape& shape, int value) {
		shared_ptr<Tensor> output = Static("const", shape);
		output->data = std::vector<uint>(1, AsUint(value));
		output->type = DataType::Int;
		return *output;
	}

	static Tensor& Constant(const Shape& shape, uint value) {
		shared_ptr<Tensor> output = Static("const", shape);
		output->data = std::vector<uint>(1, value);
		output->type = DataType::Uint;
		return *output;
	}

	static Tensor& Constant(const Shape& shape, float* data) {
		shared_ptr<Tensor> output = Static("const_memory", shape);
		uint data_count = output->Size();
		for (int i = 0; i < data_count; i++) {
			output->data.push_back(AsUint(data[i]));
		}
		return *output;
	}

	static Tensor& Input(const Shape& shape) {
		shared_ptr<Tensor> output = Static("input_memory", shape);
		output->type = DataType::MemoryRef;
		return *output;
	}

	static Tensor& Index(const Shape& shape, int dim) {
		shared_ptr<Tensor> output = Static("dim_id", shape);
		output->data = std::vector<uint>(1, dim);
		output->type = DataType::Int;
		return *output;
	}

	static Tensor& Load(const Tensor& tensor,
	                    const std::vector<const Tensor*>& indices) {
		return *IndexedOp("load", indices, &tensor);
	}

	static void Store(const Tensor& tensor, const Tensor& value,
	                  const std::vector<const Tensor*>& indices) {
		IndexedOp("store", indices, &tensor, &value);
	}

	static void ScatterAdd(const Tensor& tensor, const Tensor& value,
	                       const std::vector<const Tensor*>& indices) {
		IndexedOp("InterlockedAdd", indices, &tensor, &value);
	}

	static void ScatterMax(const Tensor& tensor, const Tensor& value,
	                       const std::vector<const Tensor*>& indices) {
		IndexedOp("InterlockedMax", indices, &tensor, &value);
	}

	static void ScatterMin(const Tensor& tensor, const Tensor& value,
	                       const std::vector<const Tensor*>& indices) {
		IndexedOp("InterlockedMin", indices, &tensor, &value);
	}

	static void ScatterOr(const Tensor& tensor, const Tensor& value,
	                      const std::vector<const Tensor*>& indices) {
		IndexedOp("InterlockedOr", indices, &tensor, &value);
	}

	static void ScatterAnd(const Tensor& tensor, const Tensor& value,
	                       const std::vector<const Tensor*>& indices) {
		IndexedOp("InterlockedAnd", indices, &tensor, &value);
	}

	static void ScatterXor(const Tensor& tensor, const Tensor& value,
	                       const std::vector<const Tensor*>& indices) {
		IndexedOp("InterlockedXor", indices, &tensor, &value);
	}

	Tensor& Index(int dim) const { return Index(this->shape, dim); }

	// destructor
	~Tensor() = default;

	static int Size(const Tensor& tensor) { return tensor.shape.GetSize(); }

	[[nodiscard]] int Size() const { return Size(*this); }

	// Method to get a value at a specific index
	[[nodiscard]] static double get(const std::vector<int>& /*indices*/) {
		return 0.0;
	}

	// Method to set a value at a specific index
	void set(const std::vector<int>& indices, double value) {}

	// Overload for operator[]
	double operator[](const std::vector<int>& indices) const {
		return get(indices);
	}

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

	Tensor& operator-() const { return *Op("neg", this); }

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

	Tensor& operator!() const { return *Op("not", this); }

	Tensor& operator~() const { return *Op("bnot", this); }

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

	static Tensor& constant(const std::vector<int>& shape, float value) {
		return *new Tensor(shape, value);
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
