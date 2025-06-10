#include "Compiler/Operation.h"

namespace TensorFrost {
OpBlock::OpBlock(Op *parent): parent_op(parent) {}

OpBlock::Iterator::Iterator(OpBlock *parent, List::iterator it)
    : parent_(parent), list_(&parent->ops), cur_(it) {}

Op* OpBlock::Iterator::operator*() const { return cur_->get(); }
Op* OpBlock::Iterator::operator->() const { return cur_->get(); }

Op * OpBlock::Iterator::get_next() const { return cur_->get(); }
Op * OpBlock::Iterator::get_prev() const {
    if (cur_ == list_->begin()) return nullptr;
    return std::prev(cur_)->get();
}

OpBlock::Iterator & OpBlock::Iterator::next() { if (cur_ != list_->end()) ++cur_; return *this; }
OpBlock::Iterator & OpBlock::Iterator::prev() { if (cur_ != list_->begin()) --cur_; return *this; }

OpBlock::Iterator& OpBlock::Iterator::insert_after(std::unique_ptr<Op> op) {
    if (op->parent_block) throw std::runtime_error("Op already belongs to a block");

    auto pos = (cur_ == list_->end()) ? list_->end() : std::next(cur_);
    cur_ = list_->insert(pos, std::move(op));   // <- after-cursor
    cur_->get()->parent_block = parent_;
    return *this;
}

OpBlock::Iterator& OpBlock::Iterator::insert_before(std::unique_ptr<Op> op) {
    if (op->parent_block) throw std::runtime_error("Op already belongs to a block");

    cur_ = list_->insert(cur_, std::move(op));   // <- before-cursor
    cur_->get()->parent_block = parent_;
    return *this;
}

OpBlock::Iterator& OpBlock::Iterator::remove() {
    if (cur_ == list_->end()) return *this; // Nothing to remove
    cur_->get()->parent_block = nullptr;    // Clear parent block reference
    cur_ = list_->erase(cur_);              // Remove and update iterator
    return *this;
}

bool OpBlock::Iterator::valid() const { return cur_ != list_->end(); }
bool OpBlock::Iterator::operator==(const Iterator &o) const { return cur_ == o.cur_; }
bool OpBlock::Iterator::operator!=(const Iterator &o) const { return cur_ != o.cur_; }

OpBlock::Iterator OpBlock::begin() { return Iterator(this, ops.begin()); }
OpBlock::Iterator OpBlock::end() { return Iterator(this, ops.end()); }

void ApplyOpTransform(OpBlock &block, const std::function<void(Op &)> &transform) {
    for (auto& op : block.ops) {
        for (auto& sub_block : op->blocks) {
            ApplyOpTransform(*sub_block, transform);
        }
        transform(*op);
    }
}

void IterateOver(OpBlock &block, const std::function<void(OpBlock::Iterator&)> &transform) {
    for (OpBlock::Iterator it = block.begin(); it.valid(); it.next()) {
        for (auto& sub_block : it->blocks) {
            IterateOver(*sub_block, transform);
        }
        transform(it);
    }
}

void ReverseIterateOver(OpBlock &block, const std::function<void(OpBlock::Iterator&)> &transform) {
    for (OpBlock::Iterator it = block.end(); it.valid(); it.prev()) {
        for (auto& sub_block : it->blocks) {
            ReverseIterateOver(*sub_block, transform);
        }
        transform(it);
    }
}

std::set<Op*> GetDependencies(std::vector<Op*> ops) {
    std::set<Op*> dependencies;
    std::function<void(Op*)> collect_dependencies = [&](Op* op) {
        if (op == nullptr || dependencies.contains(op)) return; // Already processed
        dependencies.insert(op);
        for (auto& input : op->args->Get(ArgType::Input)->inputs) {
            collect_dependencies(input->from);
        }
        for (auto& input : op->args->Get(ArgType::Index)->inputs) {
            collect_dependencies(input->from);
        }
        collect_dependencies(op->parent_block->parent_op); // Collect dependencies of the parent op
    };
    for (Op* op : ops) {
        collect_dependencies(op);
    }
    return dependencies;
}
}
