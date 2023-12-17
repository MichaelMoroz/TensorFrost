#pragma once

#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace TensorFrost {

using namespace std;

enum class DataType {
	Float,
	Uint,
	Int,
	Bool,
	None,
};

enum class OpType {
	Operator,
	Function,
	Keyword,
	Load,
	Store,
	Set,
	Scatter,
	Loop,
	Conditional,
	Special,
	Variable,
	Memory,
	UnaryOperator,
	TypeCast,
	TypeReinterpret,
	DimensionIndex,
	Constant,
};

using DataTypeList = vector<DataType>;

DataTypeList Types(initializer_list<DataType> elements);

class Operation {
public:
	string name_;
	float cost_ = 0.0F;
	vector<pair<vector<DataType>, DataType>> overloads_;
	string code_;
	OpType op_type_;


	Operation(string name, initializer_list<string> oloads, float cost,
	          string code = "", OpType op_type = OpType::Function)
	    : name_(std::move(name)), op_type_(op_type) {
		if (code.empty()) {
			code = name_;
		}

		code_ = code;
		cost_ = cost;

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

	[[nodiscard]] float GetCost() const { return cost_; }

	[[nodiscard]] OpType GetOpType() const { return op_type_; }

	[[nodiscard]] string GetName() const { return name_; }

	[[nodiscard]] vector<pair<vector<DataType>, DataType>> GetOverloads() const {
		return overloads_;
	}

	[[nodiscard]] size_t GetInputCount() const {
		return overloads_[0].first.size();
	}

	[[nodiscard]] bool IsInputValid(const vector<DataType>& input_types) const {
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

	[[nodiscard]] bool IsOutputValid(const DataType& output_type) const {
		for (const auto& overload : overloads_) {
			if (overload.second == output_type) {
				return true;
			}
		}
		return false;
	}

	[[nodiscard]] DataType GetOutputType(
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

	[[nodiscard]] bool IsOperator() const { return op_type_ == OpType::Operator; }

	[[nodiscard]] string GetCode() const { return code_; }

	[[nodiscard]] string GenerateOpString(const vector<string>& arguments) const;
	[[nodiscard]] string GenerateLine(const string& var_name,
	                                  const vector<string>& arguments,
	                                  const vector<DataType>& input_types) const;
};

const Operation& FindOperation(const string& name);

string DataTypeToString(DataType type);

string RemoveSpaces(string str);

}  // namespace TensorFrost
