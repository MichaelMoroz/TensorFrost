#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <map>
#include <initializer_list>
#include <stdexcept>

namespace TensorFrost {

using namespace std;

enum class DataType {
	Float,
	Uint,
	Int,
	Bool,
	MemoryRef,
	None,
};

using DataTypeList = vector<DataType>;

DataTypeList Types(initializer_list<DataType> elements);

class Operation {
 private:
	string name_;
	vector<pair<vector<DataType>, DataType>> overloads_;
	string code_;
	bool is_operator_ = false;

 public:
	Operation(string name,
	          initializer_list<string> oloads, string code = "", bool is_operator = false)
	    : name_(std::move(name)), is_operator_(is_operator) 
	{
		if(code.empty())
		{
			code = name_;
		}

		code_ = code;

		// parse the overloads
		// example: "ff_f" means two floats in, one float out, "buf_f" means a bool, uint, float in, float out
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
					case 'm':
						parsed_type = DataType::MemoryRef;
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

			overloads_.push_back({inputs, output});
		}
	}

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

	DataType GetOutputType(const vector<DataType>& input_types) const {
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

	[[nodiscard]] bool IsOperator() const { return is_operator_; }

	[[nodiscard]] string GetCode() const { return code_; }

	string GenerateOpString(const vector<string>& arguments) const;
	string GenerateLine(const string& var_name, const vector<string>& arguments, const vector<DataType>& input_types) const;
};

const Operation& FindOperation(const string& name);

string DataTypeToString(DataType type);

}  // namespace TensorFrost
