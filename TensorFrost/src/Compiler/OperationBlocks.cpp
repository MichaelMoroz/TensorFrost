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

OpBlock::Iterator& OpBlock::Iterator::move_before(Iterator other)
{
    auto pos = cur_;
    list_->splice(pos, *other.list_, other.cur_);
    cur_ = std::prev(pos);
    cur_->get()->parent_block = parent_;
    return *this;
}

OpBlock::Iterator& OpBlock::Iterator::move_range_before(Iterator other_start, Iterator other_end)
{
    auto pos = cur_;
    for(auto it = other_start.cur_; it != other_end.cur_; ++it)
        it->get()->parent_block = parent_;
    list_->splice(pos, *other_start.list_, other_start.cur_, other_end.cur_);
    cur_ = std::prev(pos);
    return *this;
}

bool OpBlock::Iterator::valid() const { return cur_ != list_->end(); }
bool OpBlock::Iterator::operator==(const Iterator &o) const { return cur_ == o.cur_; }
bool OpBlock::Iterator::operator!=(const Iterator &o) const { return cur_ != o.cur_; }

OpBlock::Iterator OpBlock::begin() { return Iterator(this, ops.begin()); }
OpBlock::Iterator OpBlock::end() { return Iterator(this, ops.end()); }

}
