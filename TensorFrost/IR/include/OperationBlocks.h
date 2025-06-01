#pragma once
#include "Operation.h"

namespace TensorFrost {

struct OpBlock {
    std::list<std::unique_ptr<Op>> ops;
    Op* append(std::unique_ptr<Op> op);
};

class OpBlockIterator {
public:
    using OpIter = std::list<std::unique_ptr<Op>>::iterator;
    using OpRevIter = std::list<std::unique_ptr<Op>>::reverse_iterator;

    struct Frame {
        OpBlock* block;
        OpIter it;
        OpIter end;
    };

    OpBlockIterator(OpBlock* root);

    Op* next();   // Move to next Op in depth-first order
    Op* prev();   // Move to previous Op in depth-first order
    bool down();  // Enter the first sub-block of current Op (if any)
    bool up();    // Exit to parent block

    Op* current() const;

private:
    std::vector<Frame> stack;
    Op* current_op;
};

}