#include "../include/Operation.h"
#include "../include/OperationArguments.h"
#include "../include/Overloads.h"

using namespace TensorFrost;

Op::Op(std::string op_name): opcode(std::move(op_name)) {
    args = std::make_unique<ArgumentManager>(this);
    type = TFTypeNone;
}

Op::Op(int value) : Op("const") {
    attributes["value"] = value;
    type = TFTypeInt32;
}

Op::Op(uint value) : Op("const") {
    attributes["value"] = value;
    type = TFTypeUint32;
}

Op::Op(float value) : Op("const") {
    attributes["value"] = value;
    type = TFTypeFloat32;
}

Op::Op(bool value) : Op(std::string("const")) {
    attributes["value"] = value;
    type = TFTypeBool32;
}
