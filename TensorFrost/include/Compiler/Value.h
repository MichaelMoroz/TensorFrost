#pragma once
#include "Operation.h"

namespace TensorFrost {

// Op wrapper class for overloaded mathematics and operations
class Value {
public:
    Op* op = nullptr;

    Value() = default;
    Value(Op* operation);
    Value(const Op* operation);
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

    bool Compare(const Value& other) const;
};

std::vector<Op*> values_to_ops(const std::vector<Value>& values);
std::vector<Value> ops_to_values(const std::vector<Op*>& ops);

struct Shape {
    std::vector<Value> dimensions;
    Shape(std::vector<Value> dims) : dimensions(std::move(dims)) {}
    Shape(std::initializer_list<Value> dims) : dimensions(dims) {}
    Shape() = default;
    Shape(const Shape& other) : dimensions(other.dimensions) {}

    void AddDimension(const Value& dim);
    void AddDimensions(const std::vector<Value>& dims);
    bool Broadcastable(const Shape& other) const;
};

Shape ComputeShape(Value x);
}
