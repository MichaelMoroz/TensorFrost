#pragma once

#include <initializer_list>
#include <unordered_map>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>

namespace TensorFrost {

using namespace std;

extern "C" {
	enum TF_Type {
		Float,
		Uint,
		Int,
		Bool,
		None,
	};
}

extern std::unordered_map<TF_Type, string> DataTypeNames;

enum class OpClass {
	Operator,
	Function,
	Keyword,
	Load,
	Store,
	Set,
	Scatter,
	Special,
	Variable,
	Memory,
	UnaryOperator,
	TernaryOperator,
	TypeCast,
	TypeReinterpret,
	DimensionIndex,
	CantSubstitute,
	Constant,
	MemoryOp,
	Static, //can not be removed or copied
	HostOnly,
	KernelOnly,
	Algorithm,
	Modifier,
	MemoryReuse,
	Gradient,
	Nondiff,
	Copy,
};

using DataTypeList = vector<TF_Type>;

DataTypeList Types(initializer_list<TF_Type> elements);

class Operation {
public:
	string name_;
	float cost_ = 0.0F;
	vector<pair<vector<TF_Type>, TF_Type>> overloads_;
	string code_;
	vector<OpClass> op_classes;

	Operation() = default;

	Operation(string name, initializer_list<string> overloads, float cost,
	          string code = "", initializer_list <OpClass> op_type = {})
	    : name_(std::move(name)){
		if (code.empty()) {
			code = name_;
		}

		code_ = code;
		cost_ = cost;

		//add op types
		for (const auto& type : op_type) {
			op_classes.push_back(type);
		}

		if (op_classes.empty()) {
			op_classes.push_back(OpClass::Function);
		}

		// parse the overloads
		// example: "ff_f" means two floats in, one float out, "buf_f" means a bool,
		// uint, float in, float out
		for (const auto& oload : overloads) {
			vector<TF_Type> inputs;
			TF_Type output = TF_Type::None;
			bool is_output = false;

			for (const auto& c : oload) {
				TF_Type parsed_type = TF_Type::None;
				switch (c) {
					case 'f':
						parsed_type = TF_Type::Float;
						break;
					case 'u':
						parsed_type = TF_Type::Uint;
						break;
					case 'i':
						parsed_type = TF_Type::Int;
						break;
					case 'b':
						parsed_type = TF_Type::Bool;
						break;
					case '_':
						is_output = true;
						break;
					default:
						throw std::runtime_error("Invalid character in overload string");
						break;
				}

				if (is_output) {
					output = parsed_type;
				} else {
					inputs.push_back(parsed_type);
				}
			}

			overloads_.emplace_back(inputs, output);
		}
	}

	bool HasAllTypes(OpClass type) const {
		return std::find(op_classes.begin(), op_classes.end(), type) != op_classes.end();
	}

	template <typename... Args>
	bool HasAllTypes(OpClass type, Args... args) const {
		return HasAllTypes(type) && HasAllTypes(args...);
	}

	bool HasAnyType(OpClass type) const {
		return HasAllTypes(type);
	}

	template <typename... Args>
	bool HasAnyType(OpClass type, Args... args) const {
		return HasAllTypes(type) || HasAnyType(args...);
	}

	float GetCost() const { return cost_; }

	string GetName() const { return name_; }

	vector<pair<vector<TF_Type>, TF_Type>> GetOverloads() const {
		return overloads_;
	}

	size_t GetInputCount() const {
		return overloads_[0].first.size();
	}

	bool IsInputValid(const vector<TF_Type>& input_types) const {
		for (const auto& overload : overloads_) {
			if (overload.first.size() != input_types.size()) {
				continue;
			}

			bool valid = true;
			for (size_t i = 0; i < input_types.size(); i++) {
				if (overload.first[i] != input_types[i]) {
					valid = false;
					break;
				}
			}

			if (valid) {
				return true;
			}
		}
		return false;
	}

	bool IsOutputValid(const TF_Type& output_type) const {
		for (const auto& overload : overloads_) {
			if (overload.second == output_type) {
				return true;
			}
		}
		return false;
	}

	TF_Type GetOutputType(
	    const vector<TF_Type>& input_types) const {
		for (const auto& overload : overloads_) {
			if (overload.first.size() != input_types.size()) {
				continue;
			}

			bool valid = true;
			for (size_t i = 0; i < input_types.size(); i++) {
				if (overload.first[i] != input_types[i]) {
					valid = false;
					break;
				}
			}

			if (valid) {
				return overload.second;
			}
		}
		throw std::runtime_error("Invalid input types for operation");
	}
};

const Operation* FindOperation(const string& name);

string DataTypeToString(TF_Type type);

string RemoveSpaces(string str);

}  // namespace TensorFrost
