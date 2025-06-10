#include "Compiler/Operation.h"
#include "Compiler/ExecutionContext.h"

using namespace std;

namespace TensorFrost {
// General function to create an Op instance in the current execution context
Op& make_op(std::string op, std::vector<Op*> ids, std::vector<Op*> args) {
    OpSpec* spec = GetOpSpec(op);
    vector<TFDataFormat> arg_types;
    for (const auto& arg : args) {
        arg_types.push_back(arg->type);
    }
    TFDataFormat output_type = spec->GetOutputType(arg_types);
    Op* op_instance = new Op(op);
    op_instance->type = output_type;
    op_instance->args->SetArguments(ArgType::Index, ids);
    op_instance->args->SetArguments(ArgType::Input, args);

    // Create blocks
    for (int i = 0; i < spec->blocks; ++i) {
        op_instance->NewBlock();
    }

    return GetContext()->Add(std::unique_ptr<Op>(op_instance));
}

Op & func_op(const std::string &name, std::vector<Op*> args) {
    return make_op(name,  {}, std::move(args));
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
//
// Op& unpack_tuple(const Op &x, int index) {
//     Op& elem = func_op("unpack_tuple_int", {as_op(x)});
//     elem.attributes["index"] = index; // Default index
//     return elem;
// }
//
// Op& vmap(std::vector<Op*> shape, std::function<void(Op&)> body) {
//     Op& par_op = func_op("vmap", std::move(shape));
//     GetContext()->BeginCursor(par_op.GetBlock().begin());
//     body(par_op);
//     GetContext()->EndCursor();
//     return par_op;
// }
//
// Op& memory(std::vector<Op*> shape, TFDataFormat type) {
//     Op& mem_op = func_op("memory", std::move(shape));
//     mem_op.type = type;
//     return mem_op;
// }
//
// Op& load_at_index(const Op& mem, std::vector<Op*> indices) {
//     return make_op("load", indices, {as_op(mem)});
// }

Value unpack_tuple(Value x, int index) {
    if (x.op->type.type != TFType::Tuple) {
        throw std::runtime_error("Cannot unpack non-tuple value");
    }
    Op& elem = func_op("unpack_tuple_int", {x.op});
    elem.attributes["index"] = index; // Default index
    return Value(&elem);
}

Value vmap(std::vector<Value> shape, std::function<void(Value)> body) {
    Op& par_op = func_op("vmap", {});
    for (const auto& dim : shape) {
        par_op.args->AddArgument(ArgType::Input, dim.op);
    }
    GetContext()->BeginCursor(par_op.GetBlock().begin());
    body(Value(&par_op));
    GetContext()->EndCursor();
    return Value(&par_op);
}
