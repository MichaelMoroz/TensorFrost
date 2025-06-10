#pragma once
#include "Operation.h"

namespace TensorFrost {
struct OpBlock {
    using List = std::list<std::unique_ptr<Op>>;
    using It   = List::iterator;

    Op* parent_op = nullptr;
    List ops;

    OpBlock(Op* parent = nullptr);

    class Iterator {
        OpBlock* parent_;
        List* list_;
        It    cur_;

    public:
        Iterator(OpBlock *parent, List::iterator it);

        Op* operator*()  const;
        Op* operator->() const;
        Op* get_next() const;
        Op* get_prev() const;

        Iterator& next();
        Iterator& prev();
        Iterator& insert_after(std::unique_ptr<Op> op);
        Iterator& insert_before(std::unique_ptr<Op> op);
        Iterator& remove();

        OpBlock* parent() const { return parent_; }

        bool valid() const;
        bool operator==(const Iterator& o) const;
        bool operator!=(const Iterator& o) const;
    };

    Iterator begin();
    Iterator end();
};

void ApplyOpTransform(OpBlock& block, const std::function<void(Op&)>& transform);
void IterateOver(OpBlock &block, const std::function<void(OpBlock::Iterator&)> &transform);
std::set<Op*> GetDependencies(std::vector<Op*> ops);
}
