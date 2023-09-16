#ifndef _OPERATIONS_H_
#define _OPERATIONS_H_

#include <vector>
#include <string>
#include <tuple>
#include <utility>

namespace TensorFrost
{

using namespace std;

enum class DataType {
    f32,
    u32,
    i32,
    b1,
    memory_ref,
    none,
};

#define dtype(x) DataType::x

using DataTypeList = vector<DataType>;

DataTypeList Types(initializer_list<DataType> elements);

class Operation {
private:
    string name;
    vector<pair<vector<DataType>, DataType>> overloads;

public:
    Operation(const string& name, initializer_list<pair<vector<DataType>, DataType>> oloads)
        : name(name), overloads(oloads) {}

    string GetName() const {
        return name;
    }

    vector<pair<vector<DataType>, DataType>> GetOverloads() const {
        return overloads;
    }

    int GetInputCount() const {
        return overloads[0].first.size();
    }

    bool IsInputValid(const vector<DataType>& input_types) const {
        for (const auto& overload : overloads) {
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

Operation FindOperation(string name);

}
#endif