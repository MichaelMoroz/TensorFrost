#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <variant>

namespace TensorFrost {

using uint = unsigned int;

struct Type {
    std::string dtype;
    size_t size = 0; // size in bytes
};

struct Value;
struct Op;
struct Arguments;

using Attribute = std::variant<int64_t, double, std::string, bool>;
using AttributeMap = std::unordered_map<std::string, Attribute>;

struct Value {
    int id;
    Type type;
    Op* producer = nullptr;
};

struct Block {
    std::list<std::unique_ptr<Op>> ops;
    Op* append(std::unique_ptr<Op> op) {
        ops.emplace_back(std::move(op));
        return ops.back().get();
    }
};

struct ExecutionContext {
    std::unique_ptr<Block> base_block;
    Block* current_block = base_block.get();
    std::vector<Block*> stack;

    void BeginBlock(Op* op) {
        stack.push_back(current_block);
        current_block = new Block();
    }

    void EndBlock() {
        if (!stack.empty()) {
            current_block = stack.back();
            stack.pop_back();
        } else {
            current_block = nullptr;
        }
    }

    void AddOp(std::unique_ptr<Op> op) {
        if (!current_block) {
            throw std::runtime_error("No current block to add operation to");
        }
        current_block->append(std::move(op));
    }
};

enum class ArgType {
    Input,
    Output,
    Memory,
    Shape,
    Count
};

struct Argument {
    ArgType type;
    Value* from_value = nullptr;
    Op* target_op = nullptr;
    int index = 0;
};

struct Op {
    static ExecutionContext* current_context;
    std::string opcode;
    std::vector<Argument> arguments;
    std::vector<Value> outputs;
    std::vector<std::unique_ptr<Block>> blocks;
    AttributeMap attributes;

    Op(std::string op_name) : opcode(std::move(op_name)) {
        if (!current_context) {
            throw std::runtime_error("No current execution context set for operation creation");
        }
        current_context->AddOp(std::make_unique<Op>(*this));
    }

    std::string operator std::string() const {
        return opcode;
    }

    Op& binary(Op& other, const std::string& op_name) {
        Op* new_op = new Op(op_name);
        Argument arg1{ArgType::Input, &outputs[0], this, 0};
        Argument arg2{ArgType::Input, &other.outputs[0], &other, 0};
        new_op->arguments.push_back(arg1);
        new_op->arguments.push_back(arg2);
        new_op->outputs.push_back(Value{0, Type{"float", 4}, new_op});
        return *new_op;
    }
};




} // namespace ir

