#include "Compiler/Operation.h"

namespace TensorFrost {
Op* OpBlock::append(std::unique_ptr<Op> op) {
    ops.emplace_back(std::move(op));
    return ops.back().get();
}

OpBlockIterator::OpBlockIterator(OpBlock* root) : current_op(nullptr) {
    if (root && !root->ops.empty()) {
        stack.push_back({root, root->ops.begin(), root->ops.end()});
        current_op = stack.back().it->get();
    }
}

Op* OpBlockIterator::current() const {
    return current_op;
}

Op* OpBlockIterator::next() {
    if (stack.empty()) return nullptr;
    // If current op has sub-blocks, go down
    if (!current_op->blocks.empty() && current_op->blocks[0] && !current_op->blocks[0]->ops.empty()) {
        OpBlock* sub = current_op->blocks[0].get();
        stack.push_back({sub, sub->ops.begin(), sub->ops.end()});
        current_op = stack.back().it->get();
        return current_op;
    }
    // Otherwise, go to next op in current block or up
    while (!stack.empty()) {
        auto& frame = stack.back();
        ++frame.it;
        if (frame.it != frame.end) {
            current_op = frame.it->get();
            return current_op;
        } else {
            stack.pop_back();
        }
    }
    current_op = nullptr;
    return nullptr;
}

Op* OpBlockIterator::prev() {
    if (stack.empty()) return nullptr;
    auto& frame = stack.back();
    if (frame.it == frame.block->ops.begin()) {
        stack.pop_back();
        if (!stack.empty()) {
            current_op = stack.back().it->get();
            return current_op;
        }
        current_op = nullptr;
        return nullptr;
    }
    --frame.it;
    // Go to the deepest last op in sub-blocks if any
    Op* op = frame.it->get();
    while (!op->blocks.empty() && op->blocks[0] && !op->blocks[0]->ops.empty()) {
        OpBlock* sub = op->blocks[0].get();
        stack.push_back({sub, --sub->ops.end(), sub->ops.end()});
        op = stack.back().it->get();
    }
    current_op = op;
    return current_op;
}

bool OpBlockIterator::down() {
    if (!current_op || current_op->blocks.empty() || !current_op->blocks[0] || current_op->blocks[0]->ops.empty())
        return false;
    OpBlock* sub = current_op->blocks[0].get();
    stack.push_back({sub, sub->ops.begin(), sub->ops.end()});
    current_op = stack.back().it->get();
    return true;
}

bool OpBlockIterator::up() {
    if (stack.size() <= 1) return false;
    stack.pop_back();
    current_op = stack.back().it->get();
    return true;
}
}