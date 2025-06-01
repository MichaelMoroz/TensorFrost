#include "Compiler/Operation.h"

namespace TensorFrost {
Op::Op(std::string op_name): opcode(std::move(op_name)) {
    args = std::make_unique<ArgumentManager>(this);
    type = TFTypeNone;
}

OpBlock* Op::NewBlock() {
    blocks.emplace_back(std::make_unique<OpBlock>(this));
    return blocks.back().get();
}
}
