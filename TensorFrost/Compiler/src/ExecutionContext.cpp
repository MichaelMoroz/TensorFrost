#include "Compiler/ExecutionContext.h"
#include "Compiler/Operation.h"
#include "Compiler/OperationBlocks.h"

namespace TensorFrost {
ExecutionContext::ExecutionContext(): base_block(std::make_unique<OpBlock>()) {
    cursor_stack.push(base_block->begin());
}

void ExecutionContext::BeginCursor(OpBlock::Iterator it) {
    cursor_stack.push(it);
}

void ExecutionContext::EndCursor() {
    if (cursor_stack.empty()) {
        throw std::runtime_error("This is the last cursor, cannot end it");
    }
    cursor_stack.pop();
}

Op& ExecutionContext::Add(std::unique_ptr<Op> op) {
    cursor_stack.top().insert_before(std::move(op));
    Op* new_op = *cursor_stack.top();
    cursor_stack.top().next(); // Move the cursor to the new op
    return *new_op;
}

Op& ExecutionContext::AddBeforeCursor(std::unique_ptr<Op> op) {
    cursor_stack.top().insert_before(std::move(op));
    return **cursor_stack.top();
}

ExecutionContext* current_context = nullptr;

void StartExecutionContext(ExecutionContext* ctx) {
    if (current_context) {
        throw std::runtime_error("Execution context already started");
    }
    if (!ctx) {
        throw std::invalid_argument("Execution context cannot be null");
    }
    current_context = ctx;
}

ExecutionContext* GetContext() {
    return current_context;
}

OpBlock* GetBaseBlock() {
    if (!current_context) {
        throw std::runtime_error("No execution context available");
    }
    return current_context->base_block.get();
}

OpBlock* GetCurrentBlock() {
    if (!current_context) {
        throw std::runtime_error("No execution context available");
    }
    return current_context->cursor_stack.top().parent();
}

void BeginCursor(OpBlock::Iterator it) {
    GetContext()->BeginCursor(it);
}

void BeginCursor(OpBlock& block) {
    GetContext()->BeginCursor(block.begin());
}

void BeginCursor(Op* op) {
    if (!op || !op->parent_block) {
        throw std::runtime_error("Op does not belong to a block");
    }
    OpBlock::Iterator it(op->parent_block, op->parent_block->ops.begin());
    // Find the iterator for the specific op
    for (; it.valid(); it.next()) {
        if (*it == op) {
            GetContext()->BeginCursor(it);
            return;
        }
    }
}

void EndCursor() {
    GetContext()->EndCursor();
}

void EndExecutionContext() {
    if (!current_context) {
        throw std::runtime_error("No execution context to end");
    }
    current_context = nullptr;
}
}