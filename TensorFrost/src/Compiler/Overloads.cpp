#include "Compiler/Operation.h"
#include "Compiler/ExecutionContext.h"
#include "Compiler/Value.h"

using namespace std;

namespace TensorFrost {
// General function to create an Op instance in the current execution context
Value make_op(std::string op, std::vector<Value> ids, std::vector<Value> args) {
    OpSpec* spec = GetOpSpec(op);
    vector<TFDataFormat> arg_types;
    for (const auto& arg : args) {
        arg_types.push_back(arg.op->type);
    }
    TFDataFormat output_type = spec->GetOutputType(arg_types);
    Op* op_instance = new Op(op);
    op_instance->type = output_type;
    op_instance->args->SetArguments(ArgType::Index, values_to_ops(ids));
    op_instance->args->SetArguments(ArgType::Input, values_to_ops(args));

    // Create blocks
    for (int i = 0; i < spec->blocks; ++i) {
        op_instance->NewBlock();
    }

    return Value(&GetContext()->Add(std::unique_ptr<Op>(op_instance)));
}

Value func_op(const std::string &name, std::vector<Value> args) {
    return make_op(name,  {}, std::move(args));
}

Value constant(Attribute value) {
    Value const_op = func_op("const");
    const_op.op->attributes["value"] = value;
    const_op.op->type = GetTypeFromAttribute(value);
    return const_op;
}

Value constant(int value) { return constant(Attribute(value)); }
Value constant(uint value) { return constant(Attribute(value)); }
Value constant(float value) { return constant(Attribute(value)); }
Value constant(bool value) { return constant(Attribute(value)); }

Value unpack_tuple(Value x, int index) {
    if (x.op->type != TFTypeTuple) {
        throw std::runtime_error("Cannot unpack non-tuple value");
    }
    Value elem = func_op("unpack_tuple_int", {x});
    elem.op->attributes["index"] = index; // Default index
    return elem;
}

Value vmap(std::vector<Value> shape, std::function<void(Value)> body) {
    Value par_op = func_op("vmap", shape);
    GetContext()->BeginCursor(par_op.op->GetBlock().begin());
    body(par_op);
    GetContext()->EndCursor();
    return par_op;
}

Value memory(std::vector<Value> shape, TFDataFormat type) {
    Value mem_op = func_op("memory", std::move(shape));
    mem_op.op->type = type;
    return mem_op;
}

Value load_at_index(Value mem, std::vector<Value> indices) {
    if (mem.op->type.type == TFType::None) {
        throw std::runtime_error("Cannot load from a None type memory");
    }
    return make_op("load", indices, {mem});
}
}
