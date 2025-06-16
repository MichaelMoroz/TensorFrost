#include "Compiler/Operation.h"
#include "Compiler/ExecutionContext.h"
#include "Compiler/Value.h"
using namespace std;

namespace TensorFrost {

Value::Value(Op* operation) : op(operation) {
    if (!op) {
        throw std::runtime_error("Value cannot be constructed with a null Op pointer");
    }
}

Value::Value(const Op *operation) {
    if (!operation) {
        throw std::runtime_error("Value cannot be constructed with a null Op pointer");
    }
    op = const_cast<Op*>(operation);
}

Value::Value(float value) : op(constant(value).op) {}
Value::Value(int value) : op(constant(value).op) {}
Value::Value(uint value) : op(constant(value).op) {}
Value::Value(bool value) : op(constant(value).op) {}


Value Value::operator+(const Value& other) const {
    return func_op("add", {op, other.op});
}
Value Value::operator-(const Value& other) const {
    return func_op("sub", {op, other.op});
}
Value Value::operator*(const Value& other) const {
    return func_op("mul", {op, other.op});
}
Value Value::operator/(const Value& other) const {
    return func_op("div", {op, other.op});
}
Value Value::operator%(const Value& other) const {
    return func_op("mod", {op, other.op});
}
Value Value::operator==(const Value& other) const {
    return func_op("eq", {op, other.op});
}
Value Value::operator!=(const Value& other) const {
    return func_op("ne", {op, other.op});
}
Value Value::operator<(const Value& other) const {
    return func_op("lt", {op, other.op});
}
Value Value::operator<=(const Value& other) const {
    return func_op("le", {op, other.op});
}
Value Value::operator>(const Value& other) const {
    return func_op("gt", {op, other.op});
}
Value Value::operator>=(const Value& other) const {
    return func_op("ge", {op, other.op});
}
Value Value::operator<<(const Value& other) const {
    return func_op("shl", {op, other.op});
}
Value Value::operator>>(const Value& other) const {
    return func_op("shr", {op, other.op});
}

Value Value::operator&&(const Value& other) const {
    return func_op("land", {op, other.op});
}
Value Value::operator||(const Value& other) const {
    return func_op("lor", {op, other.op});
}
Value Value::operator!() const {
    return func_op("lnot", {op});
}

Value Value::operator-() const {
    return func_op("neg", {op});
}
Value Value::operator+() const {
    return func_op("pos", {op});
}
Value Value::operator~() const {
    return func_op("not", {op});
}

bool Value::Compare(const Value &other) const {
    if(op == other.op) return true;
    return op->Compare(*other.op);
}

Value Value::operator[](int index) const {
    return unpack_tuple(*this, index);
}
Value Value::operator[](const std::vector<Value>& indices) const {
    return load_at_index(*this, indices);
}

std::vector<Op*> values_to_ops(const std::vector<Value>& values) {
    std::vector<Op*> ops;
    ops.reserve(values.size());
    for (const auto& value : values) {
        if (value.op) {
            ops.push_back(value.op);
        } else {
            throw std::runtime_error("Value contains a null Op pointer");
        }
    }
    return ops;
}

std::vector<Value> ops_to_values(const std::vector<Op*>& ops) {
    std::vector<Value> values;
    values.reserve(ops.size());
    for (const auto& op : ops) {
        if (op) {
            values.emplace_back(op);
        } else {
            throw std::runtime_error("Op pointer in vector is null");
        }
    }
    return values;
}

void Shape::AddDimension(const Value &dim) {
    dimensions.push_back(dim);
}

void Shape::AddDimensions(const std::vector<Value> &dims) {
    dimensions.insert(dimensions.end(), dims.begin(), dims.end());
}

bool Shape::Broadcastable(const Shape &other) const {
    size_t size = other.dimensions.size();
    if (dimensions.size() < size) {
        throw std::runtime_error("Other shape has more dimensions than this shape");
    }
    for (size_t i = 0; i < size; ++i) {
        if (!dimensions[i].Compare(other.dimensions[i])) {
            return false;
        }
    }
    return true;
}

Shape ComputeShape(Value x) {
    Shape shape;
    std::vector<Op*> parents;
    Op* current = x.op;
    while(current) {
        parents.push_back(current);
        current = current->parent_block->parent_op;
    }
    std::reverse(parents.begin(), parents.end());
    for (const auto& parent : parents) {
        OpSpec* spec = GetOpSpec(parent->opcode);
        if(spec->props.contains(OpProp::HasShape)) {
            shape.AddDimensions(ops_to_values(parent->args->GetInputs(ArgType::Input)));
        }
    }
    return shape;
}

} // namespace TensorFrost

