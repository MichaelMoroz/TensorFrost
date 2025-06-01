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
    op_instance->args->SetArguments(ArgType::Shape, shape);
    return GetContext()->AddOp(std::unique_ptr<Op>(op_instance));
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
}