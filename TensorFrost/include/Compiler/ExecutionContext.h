#pragma once

#include "Common.h"

namespace TensorFrost {

struct ExecutionContext {
    std::unique_ptr<OpBlock> base_block;
    OpBlock* current_block;
    std::vector<OpBlock*> stack;

    ExecutionContext();

    void BeginBlock(Op* op);
    void EndBlock();

    Op &AddOp(std::unique_ptr<Op> op);
};

void StartExecutionContext();
ExecutionContext* GetContext();
void EndExecutionContext();

}
