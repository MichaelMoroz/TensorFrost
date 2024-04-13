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
	enum DataType {
		Float,
		Uint,
		Int,
		Bool,
		None,
	};
}

extern std::unordered_map<DataType, string> DataTypeNames;

enum class OpType {
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
	Constant,
	MemoryOp,
	Static, //can not be removed or copied
	HostOnly,
	KernelOnly,
	Algorithm,
	Modifier,
};

using DataTypeList = vector<DataType>;

DataTypeList Types(initializer_list<DataType> elements);

class Operation {
public:
	string name_;
	float cost_ = 0.0F;
	vector<pair<vector<DataType>, DataType>> overloads_;
	string code_;
	vector<OpType> op_types_;

	Operation() = default;

	Operation(string name, initializer_list<string> oloads, float cost,
	          string code = "", initializer_list <OpType> op_type = {})
	    : name_(std::move(name)){
		if (code.empty()) {
			code = name_;
		}

		code_ = code;
		cost_ = cost;

		//add op types
		for (const auto& type : op_type) {
			op_types_.push_back(type);
		}

		if (op_types_.empty()) {
			op_types_.push_back(OpType::Function);
		}

		// parse the overloads
		// example: "ff_f" means two floats in, one float out, "buf_f" means a bool,
		// uint, float in, float out
		for (const auto& oload : oloads) {
			vector<DataType> inputs;
			DataType output = DataType::None;
			bool is_output = false;

			for (const auto& c : oload) {
				DataType parsed_type = DataType::None;
				switch (c) {
					case 'f':
						parsed_type = DataType::Float;
						break;
					case 'u':
						parsed_type = DataType::Uint;
						break;
					case 'i':
						parsed_type = DataType::Int;
						break;
					case 'b':
						parsed_type = DataType::Bool;
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

	bool HasAllTypes(OpType type) const {
		return std::find(op_types_.begin(), op_types_.end(), type) != op_types_.end();
	}

	template <typename... Args>
	bool HasAllTypes(OpType type, Args... args) const {
		return HasAllTypes(type) && HasAllTypes(args...);
	}

	bool HasAnyType(OpType type) const {
		return HasAllTypes(type);
	}

	template <typename... Args>
	bool HasAnyType(OpType type, Args... args) const {
		return HasAllTypes(type) || HasAnyType(args...);
	}

	float GetCost() const { return cost_; }

	string GetName() const { return name_; }

	vector<pair<vector<DataType>, DataType>> GetOverloads() const {
		return overloads_;
	}

	size_t GetInputCount() const {
		return overloads_[0].first.size();
	}

	bool IsInputValid(const vector<DataType>& input_types) const {
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

	bool IsOutputValid(const DataType& output_type) const {
		for (const auto& overload : overloads_) {
			if (overload.second == output_type) {
				return true;
			}
		}
		return false;
	}

	DataType GetOutputType(
	    const vector<DataType>& input_types) const {
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

string DataTypeToString(DataType type);

string RemoveSpaces(string str);

}  // namespace TensorFrost
