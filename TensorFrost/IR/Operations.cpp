#pragma once

#include<../../TensorFrost/IR/Operations.h>

namespace TensorFrost
{
    vector<pair<string, vector<Operation>>> operations = {
        {"operators", {
            Operation("add", {{Types({dtype(f32), dtype(f32)}), dtype(f32)}, {Types({dtype(u32), dtype(u32)}), dtype(u32)}, {Types({dtype(i32), dtype(i32)}), dtype(i32)}}),
            Operation("sub", {{Types({dtype(f32), dtype(f32)}), dtype(f32)}, {Types({dtype(u32), dtype(u32)}), dtype(u32)}, {Types({dtype(i32), dtype(i32)}), dtype(i32)}}),
            Operation("mul", {{Types({dtype(f32), dtype(f32)}), dtype(f32)}, {Types({dtype(u32), dtype(u32)}), dtype(u32)}, {Types({dtype(i32), dtype(i32)}), dtype(i32)}}),
            Operation("div", {{Types({dtype(f32), dtype(f32)}), dtype(f32)}, {Types({dtype(u32), dtype(u32)}), dtype(u32)}, {Types({dtype(i32), dtype(i32)}), dtype(i32)}}),
            Operation("mod", {{Types({dtype(u32), dtype(u32)}), dtype(u32)}, {Types({dtype(i32), dtype(i32)}), dtype(i32)}}),
            Operation("lshift", {{Types({dtype(u32), dtype(u32)}), dtype(u32)}, {Types({dtype(i32), dtype(i32)}), dtype(i32)}}),
            Operation("rshift", {{Types({dtype(u32), dtype(u32)}), dtype(u32)}, {Types({dtype(i32), dtype(i32)}), dtype(i32)}}),
            Operation("and", {{Types({dtype(u32), dtype(u32)}), dtype(u32)}, {Types({dtype(i32), dtype(i32)}), dtype(i32)}}),
            Operation("or", {{Types({dtype(u32), dtype(u32)}), dtype(u32)}, {Types({dtype(i32), dtype(i32)}), dtype(i32)}}),
            Operation("xor", {{Types({dtype(u32), dtype(u32)}), dtype(u32)}, {Types({dtype(i32), dtype(i32)}), dtype(i32)}}),
            Operation("eq", {{Types({dtype(f32), dtype(f32)}), dtype(b1)}, {Types({dtype(u32), dtype(u32)}), dtype(b1)}, {Types({dtype(i32), dtype(i32)}), dtype(b1)}}),
            Operation("neq", {{Types({dtype(f32), dtype(f32)}), dtype(b1)}, {Types({dtype(u32), dtype(u32)}), dtype(b1)}, {Types({dtype(i32), dtype(i32)}), dtype(b1)}}),
            Operation("lt", {{Types({dtype(f32), dtype(f32)}), dtype(b1)}, {Types({dtype(u32), dtype(u32)}), dtype(b1)}, {Types({dtype(i32), dtype(i32)}), dtype(b1)}}),
            Operation("lte", {{Types({dtype(f32), dtype(f32)}), dtype(b1)}, {Types({dtype(u32), dtype(u32)}), dtype(b1)}, {Types({dtype(i32), dtype(i32)}), dtype(b1)}}),
            Operation("gt", {{Types({dtype(f32), dtype(f32)}), dtype(b1)}, {Types({dtype(u32), dtype(u32)}), dtype(b1)}, {Types({dtype(i32), dtype(i32)}), dtype(b1)}}),
            Operation("gte", {{Types({dtype(f32), dtype(f32)}), dtype(b1)}, {Types({dtype(u32), dtype(u32)}), dtype(b1)}, {Types({dtype(i32), dtype(i32)}), dtype(b1)}})
        }},
        {"unary", {
            Operation("uint", {{Types({dtype(f32)}), dtype(u32)}, {Types({dtype(u32)}), dtype(u32)}, {Types({dtype(i32)}), dtype(u32)}}),
            Operation("int", {{Types({dtype(f32)}), dtype(i32)}, {Types({dtype(u32)}), dtype(i32)}, {Types({dtype(i32)}), dtype(i32)}}),
            Operation("float", {{Types({dtype(f32)}), dtype(f32)}, {Types({dtype(u32)}), dtype(f32)}, {Types({dtype(i32)}), dtype(f32)}}),
            Operation("min", {{Types({dtype(f32), dtype(f32)}), dtype(f32)}, {Types({dtype(u32), dtype(u32)}), dtype(u32)}, {Types({dtype(i32), dtype(i32)}), dtype(i32)}}),
            Operation("max", {{Types({dtype(f32), dtype(f32)}), dtype(f32)}, {Types({dtype(u32), dtype(u32)}), dtype(u32)}, {Types({dtype(i32), dtype(i32)}), dtype(i32)}}),
            Operation("abs", {{Types({dtype(f32)}), dtype(f32)}, {Types({dtype(u32)}), dtype(u32)}, {Types({dtype(i32)}), dtype(i32)}}),
            Operation("sign", {{Types({dtype(f32)}), dtype(f32)}, {Types({dtype(u32)}), dtype(u32)}, {Types({dtype(i32)}), dtype(i32)}}),
            Operation("ceil", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("floor", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("round", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("frac", {{Types({dtype(f32)}), dtype(f32)}}),
        }},
        {"unary math", {
            Operation("exp", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("exp2", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("log", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("log2", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("sqrt", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("rsqrt", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("rcp", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("sin", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("cos", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("tan", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("asin", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("acos", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("atan", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("sinh", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("cosh", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("tanh", {{Types({dtype(f32)}), dtype(f32)}}),
            Operation("pcg", {{Types({dtype(u32)}), dtype(u32)}}),
            Operation("pcgf", {{Types({dtype(u32)}), dtype(f32)}}),
        }},
        {"binary", {
            Operation("pow", {{Types({dtype(f32), dtype(f32)}), dtype(f32)}}),
            Operation("atan2", {{Types({dtype(f32), dtype(f32)}), dtype(f32)}}),
            Operation("mod", {{Types({dtype(f32), dtype(f32)}), dtype(f32)}}),
            Operation("step", {{Types({dtype(f32), dtype(f32)}), dtype(f32)}}),
        }},
        {"ternary", {
            Operation("clamp", {{Types({dtype(f32), dtype(f32), dtype(f32)}), dtype(f32)}}),
            Operation("lerp", {{Types({dtype(f32), dtype(f32), dtype(f32)}), dtype(f32)}}),
            Operation("fma", {{Types({dtype(f32), dtype(f32), dtype(f32)}), dtype(f32)}}),
            Operation("ternary", {{Types({dtype(b1), dtype(f32), dtype(f32)}), dtype(f32)}, {Types({dtype(b1), dtype(u32), dtype(u32)}), dtype(u32)}, {Types({dtype(b1), dtype(i32), dtype(i32)}), dtype(i32)}}),
        }},
        {"memory", {
            Operation("load", {{Types({dtype(memory_ref), dtype(u32)}), dtype(f32)}, {Types({dtype(memory_ref), dtype(u32)}), dtype(u32)}, {Types({dtype(memory_ref), dtype(u32)}), dtype(i32)}}),
            Operation("store", {{Types({dtype(memory_ref), dtype(u32), dtype(u32)}), dtype(none)}, {Types({dtype(memory_ref), dtype(u32), dtype(u32)}), dtype(none)}, {Types({dtype(memory_ref), dtype(u32), dtype(u32)}), dtype(none)}}),
            Operation("const_memory", {{Types({}), dtype(memory_ref)}})
        }},
        {"atomic", {
            Operation("InterlockedAdd", {{Types({dtype(memory_ref), dtype(u32), dtype(u32)}), dtype(none)}, {Types({dtype(memory_ref), dtype(i32), dtype(u32)}), dtype(none)}, {Types({dtype(memory_ref), dtype(f32), dtype(u32)}), dtype(none)}}),
            Operation("InterlockedMin", {{Types({dtype(memory_ref), dtype(u32), dtype(u32)}), dtype(none)}, {Types({dtype(memory_ref), dtype(i32), dtype(u32)}), dtype(none)}, {Types({dtype(memory_ref), dtype(f32), dtype(u32)}), dtype(none)}}),
            Operation("InterlockedMax", {{Types({dtype(memory_ref), dtype(u32), dtype(u32)}), dtype(none)}, {Types({dtype(memory_ref), dtype(i32), dtype(u32)}), dtype(none)}, {Types({dtype(memory_ref), dtype(f32), dtype(u32)}), dtype(none)}}),
            Operation("InterlockedAnd", {{Types({dtype(memory_ref), dtype(u32), dtype(u32)}), dtype(none)}}),
            Operation("InterlockedOr",  {{Types({dtype(memory_ref), dtype(u32), dtype(u32)}), dtype(none)}}),
            Operation("InterlockedXor", {{Types({dtype(memory_ref), dtype(u32), dtype(u32)}), dtype(none)}})
        }},
        {"variables", {
            Operation("variable", {{Types({}), dtype(u32)}, {Types({}), dtype(i32)}, {Types({}), dtype(f32)}}),
            Operation("const", {{Types({}), dtype(u32)}, {Types({}), dtype(i32)}, {Types({}), dtype(f32)}}),
            Operation("dim_id", {{Types({}), dtype(u32)}}),
            Operation("thread_id", {{Types({}), dtype(u32)}}),
            Operation("group_thread_id", {{Types({}), dtype(u32)}}),
            Operation("group_id", {{Types({}), dtype(u32)}}),
            Operation("group_count", {{Types({}), dtype(u32)}}),
            Operation("thread_count", {{Types({}), dtype(u32)}})
        }},
        {"control flow", {
            Operation("loop", {{Types({dtype(u32), dtype(u32), dtype(u32)}), dtype(none)}}),
            Operation("if", {{Types({dtype(b1)}), dtype(none)}}),
            Operation("break", {{Types({}), dtype(none)}}),
            Operation("continue", {{Types({}), dtype(none)}}),
            Operation("return", {{Types({}), dtype(none)}}),
            Operation("GroupMemoryBarrierWithGroupSync", {{Types({}), dtype(none)}})
        }}
    };

    DataTypeList Types(initializer_list<DataType> elements) {
        return DataTypeList(elements);
    }

    Operation FindOperation(string name)
    {
        for (auto& category : operations) {
            for (auto& op : category.second) {
                if (op.GetName() == name) {
                    return op;
                }
            }
        }
        return Operation("dtype(none)", {});
    }
}
