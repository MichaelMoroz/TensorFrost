#include "Compiler/Operation.h"
#include "Compiler/ExecutionContext.h"
#include "Compiler/Value.h"
#include "Compiler/Printer.h"

using namespace std;

namespace TensorFrost {

// General function to create an Op instance in the current execution context
std::pair<Op*, OpSpec*> create_op(std::string op, const Values& args, TFDataFormat output_type) {
    OpSpec* spec = GetOpSpec(op);
    vector<TFDataFormat> arg_types;
    for (const auto& arg : args) {
        arg_types.push_back(arg.op->type);
    }
    if (output_type == TFUnknown) {
        output_type = spec->GetOutputType(arg_types);
    }
    if (output_type == TFUnknown) {
        throw std::runtime_error("Cannot determine output type for operation '" + op + "'");
    }
    Op* op_instance = new Op(op);
    op_instance->type = output_type;
    op_instance->args->SetArguments(args);

    // Create blocks
    for (int i = 0; i < spec->blocks; ++i) {
        op_instance->NewBlock();
    }
    op_instance = &GetContext()->Add(std::unique_ptr<Op>(op_instance));
    Shape shape = ComputeShape(Value(op_instance));

    bool valid = spec->IsValid(arg_types, output_type);
    if (!valid) {
        throw std::runtime_error("Invalid operation '" + op + "' with arguments: " +
                                 PrintArray(TransformVector(values_to_ops(args), PrintOp), "[", "]", ", \n"));
    }
    return {op_instance, spec};
}

Value value_op(std::string op, Values args, TFDataFormat output_type) {
    auto [op_instance, spec] = create_op(op, args, output_type);
    if (spec->calc_tuple) throw std::runtime_error("Make op only creates single output operations, use calc_tuple for multi-output ops");
    return Value(op_instance);
}

Values tuple_op(std::string op, Values args, TFDataFormat output_type) {
    auto [op_instance, spec] = create_op(op, args, output_type);
    if (!spec->calc_tuple) throw std::runtime_error("Make tuple op only works for operations with multiple outputs");
    return spec->calc_tuple(op_instance, args);
}

Value constant(Attribute value) {
    Value const_op = value_op("const", {}, GetTypeFromAttribute(value));
    const_op.op->attributes["value"] = value;
    return const_op;
}

Value constant(int value) { return constant(Attribute(value)); }
Value constant(uint value) { return constant(Attribute(value)); }
Value constant(float value) { return constant(Attribute(value)); }
Value constant(bool value) { return constant(Attribute(value)); }

Value get_output(Value x, int index) {
    return Value(x.op, index);
}

void vmap(Values shape, std::function<void(Values)> body) {
    Values indices = tuple_op("vmap", shape);
    GetContext()->BeginCursor(indices[0].op->GetBlock().begin());
    body(indices);
    GetContext()->EndCursor();
}

Value memory(Values shape, TFDataFormat type) {
    return value_op("memory", std::move(shape), type);
}

Value load_at_index(Value mem, Values indices) {
    return value_op("load", ConcatVectors({mem}, indices));
}

void if_cond(Value cond, std::function<void()> body_true, std::function<void()> body_false) {
    Value if_op = value_op("if_cond", {cond});
    GetContext()->BeginCursor(if_op.op->GetBlock(0).begin());
    body_true();
    GetContext()->EndCursor();
    if (body_false) {
        GetContext()->BeginCursor(if_op.op->GetBlock(1).begin());
        body_false();
        GetContext()->EndCursor();
    }
}

Value loop(Value start, Value end, Value step, std::function<void(Value)> body) {
    Value loop_op = value_op("loop", {start, end, step});
    GetContext()->BeginCursor(loop_op.op->GetBlock().begin());
    body(loop_op);
    GetContext()->EndCursor();
    return loop_op;
}

Value phi(Values inputs) {
    return value_op("phi", inputs);
}

}
