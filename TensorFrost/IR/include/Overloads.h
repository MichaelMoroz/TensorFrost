#pragma once
#include "IR.h"

namespace TensorFrost {
using ArgValue = std::variant<uint, int, float, double, bool, Value&>;

// Arithmetic operations
Value& operator+(ArgValue& a, ArgValue& b);
Value& operator-(ArgValue& a, ArgValue& b);
Value& operator*(ArgValue& a, ArgValue& b);
Value& operator/(ArgValue& a, ArgValue& b);
Value& operator%(ArgValue& a, ArgValue& b);

// Bitwise operations
Value& operator&(ArgValue& a, ArgValue& b);
Value& operator|(ArgValue& a, ArgValue& b);
Value& operator^(ArgValue& a, ArgValue& b);
Value& operator<<(ArgValue& a, ArgValue& b);
Value& operator>>(ArgValue& a, ArgValue& b);
Value& operator~(ArgValue& a);

// Comparison operations
Value& operator==(ArgValue& a, ArgValue& b);
Value& operator!=(ArgValue& a, ArgValue& b);
Value& operator<(ArgValue& a, ArgValue& b);
Value& operator<=(ArgValue& a, ArgValue& b);
Value& operator>(ArgValue& a, ArgValue& b);
Value& operator>=(ArgValue& a, ArgValue& b);

// Logical operations
Value& operator&&(ArgValue& a, ArgValue& b);
Value& operator||(ArgValue& a, ArgValue& b);
Value& operator!(ArgValue& a);

// Increment and decrement operations
Value& operator++(ArgValue& a);
Value& operator--(ArgValue& a);

// Mathematical functions
Value& abs(ArgValue& a);
Value& sqrt(ArgValue& a);
Value& pow(ArgValue& base, ArgValue& exponent);
Value& sin(ArgValue& a);
Value& cos(ArgValue& a);
Value& tan(ArgValue& a);
Value& log(ArgValue& a);
Value& exp(ArgValue& a);
Value& min(ArgValue& a, ArgValue& b);
Value& max(ArgValue& a, ArgValue& b);
Value& clamp(ArgValue& value, ArgValue& min_val, ArgValue& max_val);
Value& round(ArgValue& a);
Value& floor(ArgValue& a);
Value& ceil(ArgValue& a);
Value& select(ArgValue& condition, ArgValue& true_value, ArgValue& false_value);









}
