#include "Operations.h"
#include <iostream>

namespace TensorFrost {

map<DataType, string> type_names = {
    {DataType::None, "void"}, {DataType::Bool, "bool"}, {DataType::Float, "float"},
    {DataType::Uint, "uint"}, {DataType::Int, "int"},
};

const vector<Operation> operations = {
    Operation("add", {"ff_f", "uu_u", "ii_i"}, 1, "+", {OpType::Operator}),
    Operation("sub", {"ff_f", "uu_u", "ii_i"}, 1, "-", {OpType::Operator}),
    Operation("mul", {"ff_f", "uu_u", "ii_i"}, 1, "*", {OpType::Operator}),
    Operation("div", {"ff_f", "uu_u", "ii_i"}, 2, "/", {OpType::Operator}),
    Operation("mod", {"ff_f", "uu_u", "ii_i"}, 4, "%", {OpType::Operator}),
    Operation("lshift", {"uu_u", "ii_i"}, 1, "<<", {OpType::Operator}),
    Operation("rshift", {"uu_u", "ii_i"}, 1, ">>", {OpType::Operator}),
    Operation("and", {"uu_u", "ii_i", "bb_b"}, 1, "&", {OpType::Operator}),
    Operation("or", {"uu_u", "ii_i", "bb_b"}, 1, "|", {OpType::Operator}),
    Operation("xor", {"uu_u", "ii_i", "bb_b"}, 1, "^", {OpType::Operator}),
    Operation("eq", {"ff_b", "uu_b", "ii_b", "bb_b"}, 1,
              "==", {OpType::Operator}),
    Operation("neq", {"ff_b", "uu_b", "ii_b", "bb_b"}, 1,
              "!=", {OpType::Operator}),
    Operation("lt", {"ff_b", "uu_b", "ii_b", "bb_b"}, 1, "<", {OpType::Operator}),
    Operation("lte",
              {"ff_b",
               "uu_b"
               "ii_b",
               "bb_b"},
              1, "<=", {OpType::Operator}),
    Operation("gt", {"ff_b", "uu_b", "ii_b", "bb_b"}, 1, ">", {OpType::Operator}),
    Operation("gte", {"ff_b", "uu_b", "ii_b", "bb_b"}, 1,
              ">=", {OpType::Operator}),
    Operation("not", {"b_b", "u_u", "i_i"}, 1, "!", {OpType::UnaryOperator}),
    Operation("neg", {"f_f", "u_u", "i_i"}, 1, "-", {OpType::UnaryOperator}),
    Operation("uint", {"f_u", "u_u", "i_u", "b_u"}, 1, "uint", {OpType::TypeCast}),
    Operation("int", {"f_i", "u_i", "i_i", "b_i"}, 1, "int", {OpType::TypeCast}),
    Operation("float", {"f_f", "u_f", "i_f", "b_f"}, 1, "float", {OpType::TypeCast}),
    Operation("bool", {"f_b", "u_b", "i_b", "b_b"}, 1, "bool", {OpType::TypeCast}),
    Operation("asuint", {"f_u", "u_u", "i_u"}, 0, "asuint",
              {OpType::TypeReinterpret}),
    Operation("asint", {"f_i", "u_i", "i_i"}, 0, "asint",
              {OpType::TypeReinterpret}),
    Operation("asfloat", {"f_f", "u_f", "i_f"}, 0, "asfloat",
              {OpType::TypeReinterpret}),
    Operation("min", {"ff_f", "uu_u", "ii_i"}, 1),
    Operation("max", {"ff_f", "uu_u", "ii_i"}, 1),
    Operation("abs", {"f_f", "u_u", "i_i"}, 1),
    Operation("sign", {"f_f", "u_u", "i_i"}, 1),
    Operation("ceil", {"f_f"}, 1),
    Operation("floor", {"f_f"}, 1),
    Operation("round", {"f_f"}, 1),
    Operation("frac", {"f_f"}, 1),
    Operation("exp", {"f_f"}, 32),
    Operation("exp2", {"f_f"}, 32),
    Operation("log", {"f_f"}, 32),
    Operation("log2", {"f_f"}, 32),
    Operation("sqrt", {"f_f"}, 4),
    Operation("rsqrt", {"f_f"}, 2),
    Operation("rcp", {"f_f"}, 2),
    Operation("sin", {"f_f"}, 2),
    Operation("cos", {"f_f"}, 2),
    Operation("tan", {"f_f"}, 2),
    Operation("asin", {"f_f"}, 8),
    Operation("acos", {"f_f"}, 8),
    Operation("atan", {"f_f"}, 8),
    Operation("sinh", {"f_f"}, 8),
    Operation("cosh", {"f_f"}, 8),
    Operation("tanh", {"f_f"}, 8),
    Operation("pcg", {"u_u"}, 32),
    Operation("pcgf", {"u_f"}, 34),
    Operation("pow", {"ff_f"}, 6),
    Operation("atan2", {"ff_f"}, 32),
    Operation("mod", {"ff_f"}, 2),
    Operation("step", {"ff_f"}, 2),
    Operation("clamp", {"fff_f", "uuu_u", "iii_i"}, 4),
    Operation("lerp", {"fff_f"}, 4),
    Operation("fma", {"fff_f"}, 1),
    Operation("ternary", {"bff_f", "buu_u", "bii_i"}, 4),
    Operation("load", {"_f", "_u", "_i"}, 128, "", {OpType::Load, OpType::MemoryOp}),
    Operation("store", {"f_", "u_", "i_"}, 128, "",
              {OpType::Store, OpType::MemoryOp, OpType::Modifier}),
    Operation("set", {"f_", "u_", "i_"}, 1, "",
              {OpType::Set, OpType::Modifier}),
    Operation("const_memory", {"_f", "_i", "_u"}, 0, "", {OpType::Memory}),
    Operation("memory", {"_f", "_i", "_u"}, 0, "", {OpType::Memory}),
    Operation("InterlockedAdd", {"u_", "i_", "f_"}, 256, "",
              {OpType::Scatter, OpType::MemoryOp, OpType::Modifier}),
    Operation("InterlockedMin", {"u_", "i_", "f_"}, 256, "",
              {OpType::Scatter, OpType::MemoryOp, OpType::Modifier}),
    Operation("InterlockedMax", {"u_", "i_", "f_"}, 256, "",
              {OpType::Scatter, OpType::MemoryOp, OpType::Modifier}),
    Operation("InterlockedAnd", {"u_", "i_"}, 256, "",
              {OpType::Scatter, OpType::MemoryOp, OpType::Modifier}),
    Operation("InterlockedOr", {"u_", "i_"}, 256, "",
              {OpType::Scatter, OpType::MemoryOp, OpType::Modifier}),
    Operation("InterlockedXor", {"u_", "i_"}, 256, "",
              {OpType::Scatter, OpType::MemoryOp, OpType::Modifier}),
    Operation("variable", {"_f", "_u", "_i"}, 1),
    Operation("const", {"_f", "_u", "_i"}, 0, "", {OpType::Constant}),
    Operation("dim_id", {"_i"}, 0, "dim", {OpType::DimensionIndex}),
    Operation("thread_id", {"_i"}, 0, "", {OpType::Variable}),
    Operation("group_thread_id", {"_i"}, 0),
    Operation("group_id", {"_i"}, 0),
    Operation("group_count", {"_i"}, 1),
    Operation("thread_count", {"_i"}, 1),
    Operation("loop_begin", {"iii_i"}, 100),
    Operation("loop_end", {"i_"}, 100),
    Operation("if_begin", {"b_"}, 1),
    Operation("if_end", {""}, 0),
    Operation("break", {""}, 0, "break", {OpType::Keyword}),
    Operation("continue", {""}, 0, "continue", {OpType::Keyword}),
    Operation("return", {""}, 0, "return", {OpType::Keyword}),
    Operation("GroupMemoryBarrierWithGroupSync", {""}, 256),
};

DataTypeList Types(initializer_list<DataType> elements) {
	return DataTypeList(elements);
}

const Operation& FindOperation(const string& name) {
	if (name == "") {
		throw runtime_error("Operation name is empty");
	}

	for (int i = 0; i < operations.size(); i++) {
		if (operations[i].name_ == name) {
			return operations[i];
		}
	}
	throw runtime_error("Operation not found: " + name);
}

string DataTypeToString(DataType type) { return type_names[type]; }

string RemoveSpaces(string str) {
	str.erase(remove(str.begin(), str.end(), ' '), str.end());
	return str;
}

}  // namespace TensorFrost
