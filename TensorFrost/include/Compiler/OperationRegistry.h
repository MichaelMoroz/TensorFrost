#pragma once
#include "Common.h"

namespace TensorFrost {

using OverloadsMap = std::unordered_map<std::vector<TFDataFormat>, TFDataFormat, VecHash<TFDataFormat>>;

enum class OpClass {
    Operator,
    UnaryOperator,
    Function,
    Copy,
    Keyword,
    Parallel,
    Variable,
    TypeCast,
    TypeReinterpret,
    Constant,
    TernaryOperator,
    Memory,
    None,
};

enum class OpProp {
    ShapeArgs,
    Load,
    Store,
    MemoryOp,
    Set,
};

using FoldFn = std::function<Attribute(AttributeVector)>;

[[noreturn]] inline void bad_arity(std::size_t expect, std::size_t got)
{
    throw std::invalid_argument("constant-fold expects " + std::to_string(expect) +
                                " operands, got " + std::to_string(got));
}

template<class F>
FoldFn make_fold1(F f) {
    return [f = std::move(f)](AttributeVector a) -> Attribute {
        if (a.size() != 1) bad_arity(1, a.size());
        return std::visit([&](auto &&x) -> Attribute {
            return f(std::forward<decltype(x)>(x));
        }, a[0]);
    };
}

template<class F>
FoldFn make_fold2(F f) {
    return [f = std::move(f)](AttributeVector a) -> Attribute {
        if (a.size() != 2) bad_arity(2, a.size());
        return std::visit([&](auto &&x, auto &&y) -> Attribute {
            return f(std::forward<decltype(x)>(x), std::forward<decltype(y)>(y));
        }, a[0], a[1]);
    };
}

template<class F>
FoldFn make_fold3(F f) {
    return [f = std::move(f)](AttributeVector a) -> Attribute {
        if (a.size() != 3) bad_arity(3, a.size());
        return std::visit([&](auto &&x, auto &&y, auto &&z) -> Attribute {
            return f(std::forward<decltype(x)>(x), std::forward<decltype(y)>(y), std::forward<decltype(z)>(z));
        }, a[0], a[1], a[2]);
    };
}

struct OpSpec {
    std::string name;
    OverloadsMap overloads;
    OpClass op_class = OpClass::None;
    std::set<OpProp> props;
    int blocks = 0;
    FoldFn constant_fold = nullptr;

    TFDataFormat GetOutputType(const std::vector<TFDataFormat>& args) const;
};

void RegisterOperation(const OpSpec& spec);
OpSpec* GetOpSpec(const std::string& name);

}