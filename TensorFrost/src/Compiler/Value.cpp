#include "Compiler/Operation.h"
#include "Compiler/ExecutionContext.h"
#include "Compiler/Value.h"
using namespace std;

namespace TensorFrost {

Value::Value(Op* operation, int from_index) : op(operation), out_index(from_index) {
    if (!op) {
        throw std::runtime_error("Value cannot be constructed with a null Op pointer");
    }
    if(from_index >= op->output_count) {
        throw std::out_of_range("Output index out of range for the operation");
    }
}

Value::Value(const Op *operation, int from_index) : out_index(from_index) {
    op = const_cast<Op*>(operation);
    if (!op) {
        throw std::runtime_error("Value cannot be constructed with a null Op pointer");
    }
    if(from_index >= op->output_count) {
        throw std::out_of_range("Output index out of range for the operation");
    }
}

Value::Value(float value) : op(constant(value).op) {}
Value::Value(int value) : op(constant(value).op) {}
Value::Value(uint value) : op(constant(value).op) {}
Value::Value(bool value) : op(constant(value).op) {}
Value::Value(const Value &other): op(other.op), out_index(other.out_index) {}

Value Value::operator+(const Value& other) const {
    return value_op("add", {op, other.op});
}
Value Value::operator-(const Value& other) const {
    return value_op("sub", {op, other.op});
}
Value Value::operator*(const Value& other) const {
    return value_op("mul", {op, other.op});
}
Value Value::operator/(const Value& other) const {
    return value_op("div", {op, other.op});
}
Value Value::operator%(const Value& other) const {
    return value_op("mod", {op, other.op});
}
Value Value::operator==(const Value& other) const {
    return value_op("eq", {op, other.op});
}
Value Value::operator!=(const Value& other) const {
    return value_op("ne", {op, other.op});
}
Value Value::operator<(const Value& other) const {
    return value_op("lt", {op, other.op});
}
Value Value::operator<=(const Value& other) const {
    return value_op("le", {op, other.op});
}
Value Value::operator>(const Value& other) const {
    return value_op("gt", {op, other.op});
}
Value Value::operator>=(const Value& other) const {
    return value_op("ge", {op, other.op});
}
Value Value::operator<<(const Value& other) const {
    return value_op("shl", {op, other.op});
}
Value Value::operator>>(const Value& other) const {
    return value_op("shr", {op, other.op});
}

Value Value::operator&&(const Value& other) const {
    return value_op("land", {op, other.op});
}
Value Value::operator||(const Value& other) const {
    return value_op("lor", {op, other.op});
}
Value Value::operator!() const {
    return value_op("lnot", {op});
}

Value Value::operator-() const {
    return value_op("neg", {op});
}
Value Value::operator+() const {
    return value_op("pos", {op});
}
Value Value::operator~() const {
    return value_op("not", {op});
}

bool Value::Compare(const Value &other) const {
    if(op == other.op) return true;
    return op->Compare(*other.op);
}

Value Value::operator[](const Values& indices) const {
    return load_at_index(*this, indices);
}

std::vector<Op*> values_to_ops(const Values& values) {
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

Values ops_to_values(const std::vector<Op*>& ops) {
    Values values;
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

void Shape::AddDimensions(const Values &dims) {
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
            shape.AddDimensions(parent->args->Inputs());
        }
    }
    return shape;
}

} // namespace TensorFrost

