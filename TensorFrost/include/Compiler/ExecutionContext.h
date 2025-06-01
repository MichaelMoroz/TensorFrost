#pragma once

#include "Common.h"
#include "Operation.h"

namespace TensorFrost {

struct ExecutionContext {
    std::unique_ptr<OpBlock> base_block;
    OpBlock::Iterator cursor;
    std::stack<OpBlock::Iterator*> stack;

    ExecutionContext();
    void BeginCursor(OpBlock::Iterator it);
    void EndCursor();

    Op &Add(std::unique_ptr<Op> op);
    Op &AddBeforeCursor(std::unique_ptr<Op> op);
};

void StartExecutionContext();
ExecutionContext* GetContext();
OpBlock* GetBaseBlock();
OpBlock* GetCurrentBlock();
void BeginCursor(OpBlock::Iterator it);
void BeginCursor(OpBlock& block);
void BeginCursor(Op* op);
void EndCursor();
void EndExecutionContext();

}
