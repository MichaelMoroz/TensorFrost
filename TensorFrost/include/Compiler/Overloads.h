#pragma once
#include "Operation.h"

namespace TensorFrost {
Op& make_op(std::string op, std::vector<Op*> mem, std::vector<Op*> ids, std::vector<Op*> args);
Op& func_op(const std::string& name, std::vector<Op*> args = {});

Op& constant(int value);
Op& constant(uint value);
Op& constant(float value);
Op& constant(bool value);

template<class T> concept Num   = std::is_arithmetic_v<std::remove_cvref_t<T>>;
template<class T> concept IsOp  = std::same_as<std::remove_cvref_t<T>, Op>;

template<Num T>
inline Op* as_op(T v) {
    using D = std::remove_cvref_t<T>;
    using Target =
        std::conditional_t<std::same_as<D,bool>,           bool,
        std::conditional_t<std::floating_point<D>,         float,
        std::conditional_t<std::unsigned_integral<D>,      unsigned int,
                                                            int>>>;
    return &constant(static_cast<Target>(v));
}
inline Op* as_op(const Op& x) { return &const_cast<Op&>(x); }

#define UNARY_OPERATOR(op, opname) \
template<class T> \
requires IsOp<T> \
inline Op& operator op(const T& x) { \
    return func_op(opname, {as_op(x)}); \
}

#define BINARY_OPERATOR(op, opname) \
template<class T, class U> \
requires (IsOp<T> || IsOp<U>) \
inline Op& operator op(const T& x, const U& y) { \
    return func_op(opname, {as_op(x), as_op(y)}); \
}

#define UNARY_FUNCTION(func, opname) \
template<class T> \
requires IsOp<T> \
inline Op& func(const T& x) { \
    return func_op(opname, {as_op(x)}); \
}

#define BINARY_FUNCTION(func, opname) \
template<class T, class U> \
requires (IsOp<T> || IsOp<U>) \
inline Op& func(const T& x, const U& y) { \
    return func_op(opname, {as_op(x), as_op(y)}); \
}

#define TERNARY_FUNCTION(func, opname) \
template<class T, class U, class V> \
requires (IsOp<T> || IsOp<U> || IsOp<V>) \
inline Op& func(const T& x, const U& y, const V& z) { \
    return func_op(opname, {as_op(x), as_op(y), as_op(z)}); \
}

UNARY_OPERATOR(+, "pos")
UNARY_OPERATOR(-, "neg")
UNARY_OPERATOR(~, "not")
UNARY_OPERATOR(!, "lnot")

BINARY_OPERATOR(+, "add")
BINARY_OPERATOR(-, "sub")
BINARY_OPERATOR(*, "mul")
BINARY_OPERATOR(/, "div")
BINARY_OPERATOR(%, "mod")
BINARY_OPERATOR(&, "and")
BINARY_OPERATOR(|, "or")
BINARY_OPERATOR(^, "xor")
BINARY_OPERATOR(<<, "lshift")
BINARY_OPERATOR(>>, "rshift")
BINARY_OPERATOR(==, "eq")
BINARY_OPERATOR(!=, "neq")
BINARY_OPERATOR(<, "lt")
BINARY_OPERATOR(<=, "lte")
BINARY_OPERATOR(>, "gt")
BINARY_OPERATOR(>=, "gte")
BINARY_OPERATOR(&&, "land")
BINARY_OPERATOR(||, "lor")

UNARY_FUNCTION(copy, "copy")
UNARY_FUNCTION(sin, "sin")
UNARY_FUNCTION(cos, "cos")
UNARY_FUNCTION(tan, "tan")
UNARY_FUNCTION(asin, "asin")
UNARY_FUNCTION(acos, "acos")
UNARY_FUNCTION(atan, "atan")
UNARY_FUNCTION(sinh, "sinh")
UNARY_FUNCTION(cosh, "cosh")
UNARY_FUNCTION(tanh, "tanh")
UNARY_FUNCTION(asinh, "asinh")
UNARY_FUNCTION(acosh, "acosh")
UNARY_FUNCTION(atanh, "atanh")
UNARY_FUNCTION(exp, "exp")
UNARY_FUNCTION(log, "log")
UNARY_FUNCTION(log2, "log2")
UNARY_FUNCTION(exp2, "exp2")
UNARY_FUNCTION(sqrt, "sqrt")
UNARY_FUNCTION(sqr, "sqr")
UNARY_FUNCTION(rsqrt, "rsqrt")
UNARY_FUNCTION(rcp, "rcp")
UNARY_FUNCTION(abs, "abs")
UNARY_FUNCTION(sign, "sign")
UNARY_FUNCTION(floor, "floor")
UNARY_FUNCTION(ceil, "ceil")
UNARY_FUNCTION(round, "round")
UNARY_FUNCTION(trunc, "trunc")
UNARY_FUNCTION(frac, "frac")
UNARY_FUNCTION(pcg, "pcg")
UNARY_FUNCTION(pcgf, "pcgf")
UNARY_FUNCTION(reversebits, "reversebits")
UNARY_FUNCTION(tofloat, "tofloat")
UNARY_FUNCTION(toint, "toint")
UNARY_FUNCTION(touint, "touint")
UNARY_FUNCTION(tobool, "tobool")
UNARY_FUNCTION(asfloat, "asfloat")
UNARY_FUNCTION(asint, "asint")
UNARY_FUNCTION(asuint, "asuint")
UNARY_FUNCTION(clamp, "clamp")

BINARY_FUNCTION(pow, "pow")
BINARY_FUNCTION(min, "min")
BINARY_FUNCTION(max, "max")
BINARY_FUNCTION(mod, "mod")
BINARY_FUNCTION(modf, "modf")
BINARY_FUNCTION(atan2, "atan2")
BINARY_FUNCTION(grad, "backwards_grad")

TERNARY_FUNCTION(lerp, "lerp")
TERNARY_FUNCTION(smoothstep, "smoothstep")
TERNARY_FUNCTION(select, "ternary")
TERNARY_FUNCTION(fma, "fma")

Op& unpack_tuple(const Op& x, int index = 0);
Op& vmap(std::vector<Op*> shape, std::function<void(Op&)> body);
Op& memory(std::vector<Op*> shape, TFDataFormat type);
Op& load_at_index(const Op& mem, std::vector<Op*> indices);
}
