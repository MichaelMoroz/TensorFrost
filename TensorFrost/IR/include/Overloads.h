#pragma once
#include "Operation.h"

namespace TensorFrost {

Op& make_op(std::string op, std::vector<Op*> mem, std::vector<Op*> ids, std::vector<Op*> args, std::vector<Op*> shape);

template <typename... Args>
Op& func_op(std::string op, const Args&... args) {
    std::vector<Op*> mem;
    std::vector<Op*> ids;
    std::vector<Op*> args_vec = {&args...};
    std::vector<Op*> shape;
    return make_op(op, mem, ids, args_vec, shape);
}

Op& constant(int value);
Op& constant(uint value);
Op& constant(float value);
Op& constant(bool value);

template<class T>
concept Num = std::is_arithmetic_v<std::remove_cvref_t<T>>;

template<Num T>
inline Op& as_op(T v)
{
    using D = std::remove_cvref_t<T>;
    using Target =
        std::conditional_t<std::same_as<D, bool>,          bool,
        std::conditional_t<std::floating_point<D>,         float,
        std::conditional_t<std::unsigned_integral<D>,      unsigned int,
                                                          int>>>;
    return constant(static_cast<Target>(v));
}

#define UNARY_OPERATOR(op, name) \
template<Num T> \
Op& operator op(const T& a) { \
    return func_op(name, as_op(a)); \
}

#define BINARY_OPERATOR(op, name) \
template<Num T, Num U> \
Op& operator op(const T& a, const U& b) { \
    return func_op(name, as_op(a), as_op(b)); \
}

#define UNARY_FUNCTION(name, opname) \
template<Num T> \
Op& name(const T& a) { \
    return func_op(opname, as_op(a)); \
}

#define BINARY_FUNCTION(name, opname) \
template<Num T, Num U> \
Op& name(const T& a, const U& b) { \
    return func_op(opname, as_op(a), as_op(b)); \
}

#define TERNARY_FUNCTION(name, opname) \
template<Num T, Num U, Num V> \
Op& name(const T& cond, const U& x, const V& y) { \
    return func_op(opname, as_op(cond), as_op(x), as_op(y)); \
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


// Arithmetic operations
Op& operator+(const Op& a, const Op& b);
Op& operator-(const Op& a, const Op& b);
Op& operator*(const Op& a, const Op& b);
Op& operator/(const Op& a, const Op& b);
Op& operator%(const Op& a, const Op& b);

// Bitwise operations
Op& operator&(const Op& a, const Op& b);
Op& operator|(const Op& a, const Op& b);
Op& operator^(const Op& a, const Op& b);
Op& operator<<(const Op& a, const Op& b);
Op& operator>>(const Op& a, const Op& b);
Op& operator~(const Op& a);

// Comparison operations
Op& operator==(const Op& a, const Op& b);
Op& operator!=(const Op& a, const Op& b);
Op& operator<(const Op& a, const Op& b);
Op& operator<=(const Op& a, const Op& b);
Op& operator>(const Op& a, const Op& b);
Op& operator>=(const Op& a, const Op& b);

// Logical operations
Op& operator&&(const Op& a, const Op& b);
Op& operator||(const Op& a, const Op& b);
Op& operator!(const Op& a);

// Increment and decrement operations
Op& operator++(const Op& a);
Op& operator--(const Op& a);

Op& operator+=(const Op& a, const Op& b);
Op& operator-=(const Op& a, const Op& b);

Op& copy(const Op& a);
Op& sin(const Op& a);
Op& cos(const Op& a);
Op& tan(const Op& a);
Op& asin(const Op& a);
Op& acos(const Op& a);
Op& atan(const Op& a);
Op& sinh(const Op& a);
Op& cosh(const Op& a);
Op& tanh(const Op& a);
Op& asinh(const Op& a);
Op& acosh(const Op& a);
Op& atanh(const Op& a);
Op& exp(const Op& a);
Op& log(const Op& a);
Op& log2(const Op& a);
Op& exp2(const Op& a);
Op& sqrt(const Op& a);
Op& sqr(const Op& a);
Op& rsqrt(const Op& a);
Op& rcp(const Op& a);
Op& abs(const Op& a);
Op& sign(const Op& a);
Op& floor(const Op& a);
Op& ceil(const Op& a);
Op& round(const Op& a);
Op& trunc(const Op& a);
Op& frac(const Op& a);

Op& pcg(const Op& a);
Op& pcgf(const Op& a);

Op& reversebits(const Op& a);

Op& tofloat(const Op& a);
Op& toint(const Op& a);
Op& touint(const Op& a);
Op& tobool(const Op& a);

Op& asfloat(const Op& a);
Op& asint(const Op& a);
Op& asuint(const Op& a);

Op& clamp(const Op& x, const Op& min, const Op& max);
Op& pow(const Op& x, const Op& y);
Op& min(const Op& x, const Op& y);
Op& max(const Op& x, const Op& y);
Op& mod(const Op& x, const Op& y);
Op& modf(const Op& x, const Op& y);
Op& atan2(const Op& x, const Op& y);
Op& grad(const Op& x, const Op& wrt);
Op& lerp(const Op& x, const Op& y, const Op& a);
Op& smoothstep(const Op& a, const Op& b, const Op& x);
Op& select(const Op& cond, const Op& x, const Op& y);
Op& fma(const Op& x, const Op& y, const Op& z);

}
