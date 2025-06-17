#pragma once
#include "Operation.h"

namespace TensorFrost {

// Op wrapper class for overloaded mathematics and manipulations
class Value {
public:
    Op* op = nullptr;
    int out_index = 0; // Index of the output value in the operation

    Value() = default;
    Value(Op* operation, int from_index = 0);
    Value(const Op* operation, int from_index = 0);
    Value(float value);
    Value(int value);
    Value(uint value);
    Value(bool value);
    Value(const Value& other);

    // indexed access
    Value operator[](const Values& indices) const;

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

std::vector<Op*> values_to_ops(const Values& values);
Values ops_to_values(const std::vector<Op*>& ops);

struct Shape {
    Values dimensions;
    Shape(Values dims) : dimensions(std::move(dims)) {}
    Shape(std::initializer_list<Value> dims) : dimensions(dims) {}
    Shape() = default;
    Shape(const Shape& other) : dimensions(other.dimensions) {}

    void AddDimension(const Value& dim);
    void AddDimensions(const Values& dims);
    bool Broadcastable(const Shape& other) const;
};

Shape ComputeShape(Value x);
}
