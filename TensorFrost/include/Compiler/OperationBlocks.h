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

        Iterator& move_before(Iterator other);
        Iterator& move_range_before(Iterator other_start, Iterator other_end);

        OpBlock* parent() const { return parent_; }

        bool valid() const;
        bool operator==(const Iterator& o) const;
        bool operator!=(const Iterator& o) const;
    };

    Iterator begin();
    Iterator end();
};

}
