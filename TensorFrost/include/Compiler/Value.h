#pragma once
#include "Operation.h"

namespace TensorFrost {

// Op wrapper class for overloaded mathematics and operations
class Value {
public:
    Op* op = nullptr;

    Value(Value& other) : op(other.op) {}
    Value(Op* operation);
    Value(float value);
    Value(int value);
    Value(uint value);
    Value(bool value);

    Value operator[](int index);
    Value operator[](std::vector<Value> indices);
};

std::vector<Op*> values_to_ops(const std::vector<Value>& values);
std::vector<Value> ops_to_values(const std::vector<Op*>& ops);
}