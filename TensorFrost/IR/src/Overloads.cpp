#include "../include/Overloads.h"
#include "../include/ExecutionContext.h"
#include "../include/OperationRegistry.h"
#include "../include/OperationArguments.h"

using namespace TensorFrost;
using namespace std;

// General function to create an Op instance in the current execution context
Op& make_op(string op, vector<Op*> mem, vector<Op*> ids, vector<Op*> args, vector<Op*> shape) {
    OpSpec* spec = GetOpSpec(op);
    vector<TFDataFormat> arg_types;
    for (const auto& arg : args) {
        arg_types.push_back(arg->type);
    }
    TFDataFormat output_type = spec->GetOutputType(arg_types);
    Op* op_instance = new Op(op);
    op_instance->type = output_type;
    op_instance->args->SetArguments(ArgType::Memory, mem);
    op_instance->args->SetArguments(ArgType::Index, ids);
    op_instance->args->SetArguments(ArgType::Input, args);
    op_instance->args->SetArguments(ArgType::Shape, shape);
    return GetContext()->AddOp(std::unique_ptr<Op>(op_instance));
}

Op& constant(int value) {
    Op& const_op = func_op("const");
    const_op.attributes["value"] = value;
    const_op.type = TFTypeInt32;
    return const_op;
}

Op& constant(uint value) {
    Op& const_op = func_op("const");
    const_op.attributes["value"] = value;
    const_op.type = TFTypeUint32;
    return const_op;
}

Op& constant(float value) {
    Op& const_op = func_op("const");
    const_op.attributes["value"] = value;
    const_op.type = TFTypeFloat32;
    return const_op;
}

Op& constant(bool value) {
    Op& const_op = func_op("const");
    const_op.attributes["value"] = value;
    const_op.type = TFTypeBool32;
    return const_op;
}

// Arithmetic operations
Op& operator+(const Op& a, const Op& b) { return func_op("add", a, b); }
Op& operator-(const Op& a, const Op& b) { return func_op("sub", a, b); }
Op& operator*(const Op& a, const Op& b) { return func_op("mul", a, b); }
Op& operator/(const Op& a, const Op& b) { return func_op("div", a, b); }
Op& operator%(const Op& a, const Op& b) { return func_op("mod", a, b); }

// Bitwise operations
Op& operator&(const Op& a, const Op& b) { return func_op("and", a, b); }
Op& operator|(const Op& a, const Op& b) { return func_op("or", a, b); }
Op& operator^(const Op& a, const Op& b) { return func_op("xor", a, b); }
Op& operator<<(const Op& a, const Op& b) { return func_op("lshift", a, b); }
Op& operator>>(const Op& a, const Op& b) { return func_op("rshift", a, b); }
Op& operator~(const Op& a) { return func_op("not", a); }

// Comparison operations
Op& operator==(const Op& a, const Op& b) { return func_op("eq", a, b); }
Op& operator!=(const Op& a, const Op& b) { return func_op("neq", a, b); }
Op& operator<(const Op& a, const Op& b) { return func_op("lt", a, b); }
Op& operator<=(const Op& a, const Op& b) { return func_op("lte", a, b); }
Op& operator>(const Op& a, const Op& b) { return func_op("gt", a, b); }
Op& operator>=(const Op& a, const Op& b) { return func_op("gte", a, b); }

// Logical operations
Op& operator&&(const Op& a, const Op& b) { return func_op("land", a, b); }
Op& operator||(const Op& a, const Op& b) { return func_op("lor", a, b); }
Op& operator!(const Op& a) { return func_op("lnot", a); }

// Assignment operations
// Op& operator+=(const Op& a, const Op& b) { return func_op("add_assign", a, b); }
// Op& operator-=(const Op& a, const Op& b) { return func_op("sub_assign", a, b); }
// Op& operator*=(const Op& a, const Op& b) { return func_op("mul_assign", a, b); }
// Op& operator/=(const Op& a, const Op& b) { return func_op("div_assign", a, b); }
// Op& operator%=(const Op& a, const Op& b) { return func_op("mod_assign", a, b); }
// Op& operator&=(const Op& a, const Op& b) { return func_op("and_assign", a, b); }
// Op& operator|=(const Op& a, const Op& b) { return func_op("or_assign", a, b); }
// Op& operator^=(const Op& a, const Op& b) { return func_op("xor_assign", a, b); }
// Op& operator<<=(const Op& a, const Op& b) { return func_op("lshift_assign", a, b); }
// Op& operator>>=(const Op& a, const Op& b) { return func_op("rshift_assign", a, b); }
// Op& operator++(const Op& a) { return a += 1; }
// Op& operator--(const Op& a) { return a -= 1; }

Op& copy(const Op& a) { return func_op("copy", a); }
Op& sin(const Op& a) { return func_op("sin", a); }
Op& cos(const Op& a) { return func_op("cos", a); }
Op& tan(const Op& a) { return func_op("tan", a); }
Op& asin(const Op& a) { return func_op("asin", a); }
Op& acos(const Op& a) { return func_op("acos", a); }
Op& atan(const Op& a) { return func_op("atan", a); }
Op& sinh(const Op& a) { return func_op("sinh", a); }
Op& cosh(const Op& a) { return func_op("cosh", a); }
Op& tanh(const Op& a) { return func_op("tanh", a); }
Op& asinh(const Op& a) { return func_op("asinh", a); }
Op& acosh(const Op& a) { return func_op("acosh", a); }
Op& atanh(const Op& a) { return func_op("atanh", a); }
Op& exp(const Op& a) { return func_op("exp", a); }
Op& log(const Op& a) { return func_op("log", a); }
Op& log2(const Op& a) { return func_op("log2", a); }
Op& exp2(const Op& a) { return func_op("exp2", a); }
Op& sqrt(const Op& a) { return func_op("sqrt", a); }
Op& sqr(const Op& a) { return func_op("sqr", a); }
Op& rsqrt(const Op& a) { return func_op("rsqrt", a); }
Op& rcp(const Op& a) { return func_op("rcp", a); }
Op& abs(const Op& a) { return func_op("abs", a); }
Op& sign(const Op& a) { return func_op("sign", a); }
Op& floor(const Op& a) { return func_op("floor", a); }
Op& ceil(const Op& a) { return func_op("ceil", a); }
Op& round(const Op& a) { return func_op("round", a); }
Op& trunc(const Op& a) { return func_op("trunc", a); }
Op& frac(const Op& a) { return func_op("frac", a); }
Op& pcg(const Op& a) { return func_op("pcg", a); }
Op& pcgf(const Op& a) { return func_op("pcgf", a); }
Op& reversebits(const Op& a) { return func_op("reversebits", a); }

Op& clamp(const Op& x, const Op& min, const Op& max) { return func_op("clamp", x, min, max); }
Op& pow(const Op& x, const Op& y) { return func_op("pow", x, y); }
Op& min(const Op& x, const Op& y) { return func_op("min", x, y); }
Op& max(const Op& x, const Op& y) { return func_op("max", x, y); }
Op& mod(const Op& x, const Op& y) { return func_op("mod", x, y); }
Op& modf(const Op& x, const Op& y) { return func_op("modf", x, y); }
Op& atan2(const Op& x, const Op& y) { return func_op("atan2", x, y); }
Op& grad(const Op& x, const Op& wrt) { return func_op("backwards_grad", x, wrt); }
Op& lerp(const Op& x, const Op& y, const Op& a) { return func_op("lerp", x, y, a); }
Op& smoothstep(const Op& a, const Op& b, const Op& x) { return func_op("smoothstep", a, b, x); }
Op& select(Op& cond, const Op& x, const Op& y) { return func_op("ternary", cond, x, y); }
Op& fma(const Op& x, const Op& y, const Op& z) { return func_op("fma", x, y, z); }

// Type conversion operations
Op& tofloat(const Op& a) { return func_op("tofloat", a); }
Op& toint(const Op& a) { return func_op("toint", a); }
Op& touint(const Op& a) { return func_op("touint", a); }
Op& tobool(const Op& a) { return func_op("tobool", a); }

Op& asfloat(const Op& a) { return func_op("asfloat", a); }
Op& asint(const Op& a) { return func_op("asint", a); }
Op& asuint(const Op& a) { return func_op("asuint", a); }