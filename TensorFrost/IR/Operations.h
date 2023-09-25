#pragma once

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
	MemoryRef,
	None,
};

using DataTypeList = vector<DataType>;

DataTypeList Types(initializer_list<DataType> elements);

class Operation {
 private:
	string name_;
	vector<pair<vector<DataType>, DataType>> overloads_;

 public:
	Operation(string name,
	          initializer_list<pair<vector<DataType>, DataType>> oloads)
	    : name_(std::move(name)), overloads_(oloads) {}

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
};

Operation FindOperation(const string& name);

}  // namespace TensorFrost
