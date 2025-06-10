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

Value::Value(float value) : op(&constant(value)) {}
Value::Value(int value) : op(&constant(value)) {}
Value::Value(uint value) : op(&constant(value)) {}
Value::Value(bool value) : op(&constant(value)) {}

Value Value::operator[](int index) {
    return unpack_tuple(*this, index);
}

Op& Op::operator[](std::vector<Op*> indices) {
    return load_at_index(*this, std::move(indices));
}

}