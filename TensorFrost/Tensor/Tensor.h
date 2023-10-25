#pragma once

#include <list>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>

#include <math.h>

#include "IR/IR.h"
#include "Utility/Utility.h"

namespace TensorFrost {

using Tensors = vector<const Tensor*>;

class Tensor {
 private:
	static void AddArguments(Arguments& arguments, const Tensors& tensors,
	                         Argument::Type type) {
		for (int i = 0; i < tensors.size(); i++) {
			arguments.emplace_back(type, tensors[i]->node, i);
		}
	}

	static void AddArguments(Arguments& arguments, const Arguments& toadd) {
		for (const auto& i : toadd) {
			arguments.push_back(i);
		}
	}

	static pair<Operation, DataType> GetOperation(const string& name, const Tensors& tensors) {
		vector<DataType> input_types = vector<DataType>();
		for (const auto& tensor : tensors) {
			input_types.push_back(tensor->type);
		}

		const Operation& operation = FindOperation(name);

		// check if input is valid
		if (!operation.IsInputValid(input_types)) {
			throw std::runtime_error("Invalid input types for operation " + name);
		}

		DataType output_type = operation.GetOutputType(input_types);

		return pair<Operation, DataType>(operation, output_type);
	}

	template <typename... Args>
	static Tensor& Op(const std::string& op, const Args*... args) {
		// convert the parameter pack to a std::vector
		Tensors tensors = {args...};

		// get the operation and output type
		pair<Operation, DataType> operation = GetOperation(op, tensors);
		DataType output_type = operation.second;

		// create argument list
		Arguments arguments = Arguments();

		AddArguments(arguments, tensors, Argument::Type::Input);
		AddArguments(arguments, tensors[0]->node->GetArguments(Argument::Type::Shape));

		auto* output = new Tensor(op, output_type, operation.first);

		// create the output tensor
		AddToGraph(output, arguments);
		return *output;
	}

	template <typename... Args>
	static Tensor& IndexedOp(const string op, const Tensors indices,
	                         const Args*... args) {
		// convert the parameter pack to a std::vector
		Tensors tensors = {args...};

		// get the operation and output type
		pair<Operation, DataType> operation = GetOperation(op, tensors);
		DataType output_type = operation.second;

		// create argument list
		Arguments arguments = Arguments();

		// add the input tensors
		AddArguments(arguments, tensors, Argument::Type::Input);

		// add the indices
		AddArguments(arguments, indices, Argument::Type::Index);

		// add the shape
		AddArguments(arguments, indices[0]->node->GetArguments(Argument::Type::Shape));

		// create the output tensor
		auto* output = new Tensor(op, output_type, operation.first);

		output->type = output_type;
		// create the output tensor
		AddToGraph(output, arguments);

		return *output;
	}

	static Tensor& Static(const string& op, const Arguments& shape, const DataType type) {
		const Operation& operation = FindOperation(op);
		// check if output is valid
		if (!operation.IsOutputValid(type)) {
			throw std::runtime_error("Invalid output type for operation " + op);
		}
		Arguments arguments = Arguments();
		AddArguments(arguments, shape);
		auto* output = new Tensor(op, type, operation);
		AddToGraph(output, arguments);
		return *output;
	}

	static Tensor& Static(const string& op, const Tensors& shape, const DataType type) {
		Arguments arguments = Arguments();
		AddArguments(arguments, shape, Argument::Type::Shape);
		return Static(op, arguments, type);
	}

	static Tensor& Static(const string& op, const DataType type) {
		return Static(op, Arguments(), type);
	}

	static IR* evaluation_context_ir_;

	static void AddToGraph(Tensor* tensor, Arguments args) {
		if (evaluation_context_ir_ == nullptr) {
			throw std::runtime_error("Evaluation context has not been set. Are you doing operations outside a TensorProgram?");
		}

		if (tensor->node != nullptr) {
			throw std::runtime_error("Tensor has already been added to the graph.");
		}

		tensor->node = evaluation_context_ir_->AddNode(tensor, args);
	}

 public:
	static void SetEvaluationContext(IR* ir) {
		if (evaluation_context_ir_ != nullptr && ir != nullptr) {
			throw std::runtime_error("Evaluation context change is forbidden.");
		}
		evaluation_context_ir_ = ir;
	}

	[[nodiscard]] string GetConstantString() const;

	string name;
	const Operation* op;
	Node* node = nullptr;
	DataType type = DataType::Float;
	std::vector<uint> data;

	// Main constructor
	Tensor(string name, DataType type, const Operation& operation) {
		this->name = std::move(name);
		this->type = type;
		this->op = &operation;
	}

	[[nodiscard]] int GetDimension() const {
		// find max dimension
		int max_dim = -1;

		for (const auto& input : node->arguments_) {
			if (input.type_ == Argument::Type::Shape) {
				max_dim = std::max(max_dim, input.index_);
			}
		}

		return max_dim + 1;
	}

	[[nodiscard]] vector<const Tensor*> GetShape() const {
		vector<const Tensor*> result = vector<const Tensor*>();
		// get max dimension
		int max_dim = -1;
		for (const auto& input : node->arguments_) {
			if (input.type_ == Argument::Type::Shape) {
				max_dim = std::max(max_dim, input.index_);
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
		for (const auto& input : node->arguments_) {
			if (input.type_ == Argument::Type::Shape) {
				result[input.index_] = input.node_->tensor_;
			}
		}
		//if there are any missing dimensions, fill them with 1
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
		for (const auto& input : node->arguments_) {
			if (input.type_ == Argument::Type::Shape) {
				max_dim = std::max(max_dim, input.index_);
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
		for (const auto& input : node->arguments_) {
			if (input.type_ == Argument::Type::Shape) {
				result[input.index_] = AsInt(input.node_->tensor_->data[0]);
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
		Tensor& output = Static("const", DataType::Float);
		output.data = std::vector<uint>(1, AsUint(value));
		return output;
	}
	static Tensor& Constant(int value) {
		Tensor& output = Static("const", DataType::Int);
		output.data = std::vector<uint>(1, AsUint(value));
		return output;
	}
	static Tensor& Constant(uint value) {
		Tensor& output = Static("const", DataType::Uint);
		output.data = std::vector<uint>(1, value);
		return output;
	}

	static Tensor& Constant(const vector<int>& shape, float* data) {
		Tensor& output = Static("const_memory", GetConstantShape(shape), DataType::Float);
		int data_count = GetSize(shape);
		for (int i = 0; i < data_count; i++) {
			output.data.push_back(AsUint(data[i]));
		}
		return output;
	}
	static Tensor& Constant(const Tensors& shape, float value) {
		Arguments arguments = Arguments();
		AddArguments(arguments, shape, Argument::Type::Shape);
		Tensor& output = Static("const", arguments, DataType::Float);
		output.data = std::vector<uint>(1, AsUint(value));
		return output;
	}
	static Tensor& Constant(const vector<int>& shape, float value) {
		return Constant(GetConstantShape(shape), value);
	}
	static Tensor& Constant(const Tensors& shape, int value) {
		Arguments arguments = Arguments();
		AddArguments(arguments, shape, Argument::Type::Shape);
		Tensor& output = Static("const", arguments, DataType::Int);
		output.data = std::vector<uint>(1, AsUint(value));
		return output;
	}
	static Tensor& Constant(const vector<int>& shape, int value) {
		return Constant(GetConstantShape(shape), value);
	}
	static Tensor& Constant(const Tensors& shape, uint value) {
		Arguments arguments = Arguments();
		AddArguments(arguments, shape, Argument::Type::Shape);
		Tensor& output = Static("const", arguments, DataType::Uint);
		output.data = std::vector<uint>(1, value);
		return output;
	}
	static Tensor& Constant(const vector<int>& shape, uint value) {
		return Constant(GetConstantShape(shape), value);
	}
	static Tensor& Input(const DataType type = DataType::Float) {
		Tensor& output = Static("input_memory", type);
		return output;
	}
	static Tensor& Input(const Tensors& shape, const DataType type = DataType::Float) {
		Tensor& output = Static("input_memory", shape, type);
		return output;
	}
	static vector<const Tensor*> GetInputShape(const vector<int>& shape) {
		Tensors result = vector<const Tensor*>();
		for (int i : shape) {
			if (i < 0) {
				result.push_back(&Input(DataType::Int));
			} else {
				result.push_back(&Constant(i));
			}
		}
		return result;
	}
	static Tensor& Input(const vector<int>& shape, const DataType type = DataType::Float) {
		return Input(GetInputShape(shape), type);
	}

	static Tensor& Index(const Tensors& shape, int dim) {
		Tensor& output = Static("dim_id", shape, DataType::Int);
		output.data = std::vector<uint>(1, dim);
		output.type = DataType::Int;
		return output;
	}
	static Tensor& Load(const Tensor& tensor, const Tensors& indices) {
		return IndexedOp("load", indices, &tensor);
	}

	[[nodiscard]] Tensor& Index(int dim) const {
		Tensor& output =
		    Static("dim_id", node->GetArguments(Argument::Type::Shape), DataType::Int);
		output.data = std::vector<uint>(1, dim);
		output.type = DataType::Int;
		return output;
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

	static Tensor& Sum(const Tensor& tensor, const int dim) {
		Tensor& res = Op("sum", &tensor);
		res.data = std::vector<uint>(1, dim);
		return res;
	}

	static void Loop(const Tensor& start, const Tensor& end, const Tensor& step,
	                 const std::function<void(const Tensor&)>& body) {
		// create the loop
		Tensor& loop = Op("loop_begin", &start, &end, &step);

		// create the body
		body(loop);
	
		// end the loop
		Op("loop_end", &loop);
	}

	// destructor
	~Tensor() = default;

	Tensor& operator-() const { return Op("neg", this); }
	Tensor& operator!() const { return Op("not", this); }
	Tensor& operator~() const { return Op("bnot", this); }

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
		return Op("band", this, &other);
	}

	Tensor& operator|(const Tensor& other) const {
		return Op("bor", this, &other);
	}

	Tensor& operator^(const Tensor& other) const {
		return Op("bxor", this, &other);
	}

	Tensor& operator<<(const Tensor& other) const {
		return Op("blshift", this, &other);
	}

	Tensor& operator>>(const Tensor& other) const {
		return Op("brshift", this, &other);
	}

	static Tensor& ifcond(const Tensor& condition, const Tensor& ifTrue,
	                      const Tensor& ifFalse) {
		return Op("cond", &condition, &ifTrue, &ifFalse);
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

	static Tensor& atan2(const Tensor& x, const Tensor& y) {
		return Op("atan2", &x, &y);
	}

	static Tensor& lerp(const Tensor& x, const Tensor& y, const Tensor& a) {
		return Op("lerp", &x, &y, &a);
	}

	static Tensor& fma(const Tensor& x, const Tensor& y, const Tensor& z) {
		return Op("fma", &x, &y, &z);
	}
};

}  // namespace TensorFrost
