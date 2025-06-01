#include "Compiler/ExecutionContext.h"
#include "Compiler/Operation.h"
#include "Compiler/OperationBlocks.h"

namespace TensorFrost {
ExecutionContext::ExecutionContext(): base_block(std::make_unique<OpBlock>()), current_block(base_block.get()) {}

void ExecutionContext::BeginBlock(Op *op) {
    stack.push_back(current_block);
    current_block = new OpBlock();
}

void ExecutionContext::EndBlock() {
    if (!stack.empty()) {
        current_block = stack.back();
        stack.pop_back();
    } else {
        throw std::runtime_error("No block to end");
    }
}

Op &ExecutionContext::AddOp(std::unique_ptr<Op> op) {
    current_block->append(std::move(op));
    return *current_block->ops.back();
}

ExecutionContext* current_context = nullptr;

void StartExecutionContext() {
    if (current_context) {
        throw std::runtime_error("Execution context already started");
    }
    current_context = new ExecutionContext();
}

ExecutionContext* GetContext() {
    return current_context;
}

void EndExecutionContext() {
    if (!current_context) {
        throw std::runtime_error("No execution context to end");
    }
    delete current_context;
    current_context = nullptr;
}
}