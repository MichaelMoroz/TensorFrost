#pragma once
#include "Operation.h"
#include "Value.h"

namespace TensorFrost {
Value make_op(std::string op, std::vector<Value> ids = {}, std::vector<Value> args = {});
Value func_op(const std::string &name, std::vector<Value> args = {});
Value constant(int value);
Value constant(uint value);
Value constant(float value);
Value constant(bool value);

Value unpack_tuple(Value x, int index = 0);
Value vmap(std::vector<Value> shape, std::function<void(Value)> body);
Value memory(std::vector<Value> shape, TFDataFormat type);
Value load_at_index(Value mem, std::vector<Value> indices);

inline Value toint(Value x) { return func_op("toint", {x}); }
inline Value tofloat(Value x) { return func_op("tofloat", {x}); }
inline Value touint(Value x) { return func_op("touint", {x}); }
inline Value tobool(Value x) { return func_op("tobool", {x}); }
}
