#include "Compiler/Operation.h"
#include "Compiler/ExecutionContext.h"

using namespace std;

namespace TensorFrost {
// General function to create an Op instance in the current execution context
Op& make_op(std::string op, std::vector<Op*> mem, std::vector<Op*> ids, std::vector<Op*> args, std::vector<Op*> shape) {
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

    if(shape.empty()) {
        shape = op_instance->args->Get(ArgType::Shape)->Inputs();
    }

    op_instance->args->SetArguments(ArgType::Shape, shape);

    // Create blocks
    for (int i = 0; i < spec->blocks; ++i) {
        op_instance->NewBlock();
    }

    return GetContext()->Add(std::unique_ptr<Op>(op_instance));
}

Op & func_op(const std::string &name, std::vector<Op *> args, std::vector<Op *> shape) {
    return make_op(name, {}, {}, std::move(args), std::move(shape));
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

Op & vmap(std::vector<Op *> shape, std::function<void(Op *)> body) {
    Op& par_op = func_op("vmap", {}, shape);
    GetContext()->BeginCursor(par_op.blocks.front()->begin());
    body(&par_op);
    GetContext()->EndCursor();
    return par_op;
}
}
