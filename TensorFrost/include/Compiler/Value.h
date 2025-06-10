#pragma once
#include "Operation.h"

namespace TensorFrost {

// Op wrapper class for overloaded mathematics and operations
class Value {
public:
    Op* op = nullptr;

    Value(Op* operation);
    Value(float value);
    Value(int value);
    Value(uint value);
    Value(bool value);
    Value(const Value& other) : op(other.op) {}

    // indexed access
    Value operator[](int index) const;
    Value operator[](const std::vector<Value>& indices) const;

    // binary ops take const ref and are const themselves
    Value operator+(const Value& other) const;
    Value operator-(const Value& other) const;
    Value operator*(const Value& other) const;
    Value operator/(const Value& other) const;
    Value operator%(const Value& other) const;
    Value operator==(const Value& other) const;
    Value operator!=(const Value& other) const;
    Value operator<(const Value& other) const;
    Value operator<=(const Value& other) const;
    Value operator>(const Value& other) const;
    Value operator>=(const Value& other) const;
    Value operator&&(const Value& other) const;
    Value operator||(const Value& other) const;
    Value operator<<(const Value& other) const;
    Value operator>>(const Value& other) const;

    // unary ops
    Value operator!() const;
    Value operator-() const;
    Value operator+() const;
    Value operator~() const;
};

std::vector<Op*> values_to_ops(const std::vector<Value>& values);
std::vector<Value> ops_to_values(const std::vector<Op*>& ops);
}