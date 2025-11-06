#pragma once
#include "Operation.h"
#include "Value.h"

namespace TensorFrost {
std::pair<Op*, OpSpec*> create_op(std::string op, const Values& args, TFDataFormat output_type  = TFUnknown);
Value value_op(std::string op, Values args = {}, TFDataFormat output_type = TFUnknown);
Values tuple_op(std::string op, Values args  = {}, TFDataFormat output_type = TFUnknown);

Value constant(int value);
Value constant(uint value);
Value constant(float value);
Value constant(bool value);

void vmap(Values shape, std::function<void(Values)> body);
Value memory(Values shape, TFDataFormat type);
Value load_at_index(Value mem, Values indices);
void if_cond(Value cond, std::function<void()> body_true, std::function<void()> body_false = nullptr);
Value loop(Value start, Value end, Value step, std::function<void(Value)> body);
Value phi(Values inputs);

inline Value toint(Value x) { return value_op("toint", {x}); }
inline Value tofloat(Value x) { return value_op("tofloat", {x}); }
inline Value touint(Value x) { return value_op("touint", {x}); }
inline Value tobool(Value x) { return value_op("tobool", {x}); }

inline Value sin(Value x) { return value_op("sin", {x}); }
inline Value cos(Value x) { return value_op("cos", {x}); }
inline Value tan(Value x) { return value_op("tan", {x}); }
}
