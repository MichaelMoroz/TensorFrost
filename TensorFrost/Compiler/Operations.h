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

#include "Utility/Utility.h"

namespace TensorFrost {

using namespace std;

extern "C" {
	enum TFType {
		Float,
		Uint,
		Int,
		Bool,
		None,
	};
}

extern std::unordered_map<TFType, string> DataTypeNames;
extern std::unordered_map<TFType, string> type_names;

//op can have only one class
enum class OpClass {
	Operator,
	UnaryOperator,
	Function,
	Copy,
	Keyword,
	DimensionIndex,
	Variable,
	TypeCast,
	TypeReinterpret,
	Constant,
	TernaryOperator,
	None,
};

//op can have multiple properties
enum class OpProp {
	Load,
	Store,
	Set,
	Scatter,
	Special,
	Memory,
	LocalMemory,
	CantSubstitute,
	MemoryOp,
	LocalMemoryOp,
	Static, //can not be removed or copied
	HostOnly,
	KernelOnly,
	Composite,
	Algorithm,
	Custom,
	Reduction,
	Scan,
	Modifier,
	MemoryReuse,
	Gradient,
	Nondiff,
	HasChildren,
	Debug,
	Count,
};

using OpProps = FlagSet<OpProp, (int)OpProp::Count>;

using DataTypeList = vector<TFType>;

DataTypeList Types(initializer_list<TFType> elements);

class Operation {
public:
	string name_;
	float cost_ = 0.0F;
	vector<pair<vector<TFType>, TFType>> overloads_;
	string code_;
	//vector<OpClass> op_classes;
	OpProps props_;
	OpClass class_;

	Operation() = default;

	Operation(string name, vector<string> overloads, float cost,
	          string code = "", initializer_list<OpProp> op_props = {},  OpClass op_class = OpClass::Function)
	    : name_(std::move(name)){
		if (code.empty()) {
			code = name_;
		}

		code_ = code;
		cost_ = cost;

		//add op types
		for (const auto& type : op_props) {
			props_.set(type);
		}

		class_ = op_class;

		// parse the overloads
		// example: "ff_f" means two floats in, one float out, "buf_f" means a bool,
		// uint, float in, float out
		for (const auto& oload : overloads) {
			vector<TFType> inputs;
			TFType output = TFType::None;
			bool is_output = false;

			for (const auto& c : oload) {
				TFType parsed_type = TFType::None;
				switch (c) {
					case 'f':
						parsed_type = TFType::Float;
						break;
					case 'u':
						parsed_type = TFType::Uint;
						break;
					case 'i':
						parsed_type = TFType::Int;
						break;
					case 'b':
						parsed_type = TFType::Bool;
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

	bool HasAllTypes(OpProp type) const {
		return props_.has(type);
	}

	template <typename... Args>
	bool HasAllTypes(OpProp type, Args... args) const {
		return HasAllTypes(type) && HasAllTypes(args...);
	}

	bool HasAnyType(OpProp type) const {
		return HasAllTypes(type);
	}

	template <typename... Args>
	bool HasAnyType(OpProp type, Args... args) const {
		return HasAllTypes(type) || HasAnyType(args...);
	}

	float GetCost() const { return cost_; }

	string GetName() const { return name_; }

	vector<pair<vector<TFType>, TFType>> GetOverloads() const {
		return overloads_;
	}

	size_t GetInputCount() const {
		return overloads_[0].first.size();
	}

	bool IsInputValid(const vector<TFType>& input_types) const {
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

	bool IsOutputValid(const TFType& output_type) const {
		for (const auto& overload : overloads_) {
			if (overload.second == output_type) {
				return true;
			}
		}
		return false;
	}

	TFType GetOutputType(
	    const vector<TFType>& input_types) const {
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

string DataTypeToString(TFType type);

string RemoveSpaces(string str);

void RegisterNewOperation(const Operation* op);

}  // namespace TensorFrost
