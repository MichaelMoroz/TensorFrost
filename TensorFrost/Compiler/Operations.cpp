#include "Operations.h"
#include <iostream>

namespace TensorFrost {

unordered_map<TFType, string> type_names = {
    {TFType::None, "void"}, {TFType::Bool, "bool"}, {TFType::Float, "float"},
    {TFType::Uint, "uint"}, {TFType::Int, "int"},
};

std::unordered_map<TFType, string> DataTypeNames = {
	{TFType::Float, "Float"}, {TFType::Uint, "Uint"},
	{TFType::Int, "Int"},     {TFType::Bool, "Bool"},
	{TFType::None, "None"},
};

const vector<Operation> operations = {
    //Scope operations
    Operation("host", {""}, 0, "", {OpProp::Static, OpProp::Special, OpProp::HostOnly, OpProp::Nondiff, OpProp::HasChildren}),
    Operation("kernel", {""}, 0, "", {OpProp::Static, OpProp::Special, OpProp::HostOnly, OpProp::Nondiff, OpProp::HasChildren}),

    //Control operations
    Operation("loop", {"iii_i"}, 100, "", {OpProp::Static, OpProp::Special, OpProp::Nondiff, OpProp::HasChildren}),
    Operation("if", {"b_"}, 100, "", {OpProp::Static, OpProp::Special, OpProp::Nondiff, OpProp::HasChildren}),
    Operation("break", {""}, 0, "break", {OpProp::Static, OpProp::Nondiff}, OpClass::Keyword),
    Operation("continue", {""}, 0, "continue", {OpProp::Static, OpProp::Nondiff}, OpClass::Keyword),
    Operation("discard", {""}, 0, "discard", {OpProp::Static, OpProp::Nondiff}, OpClass::Keyword), //discard current thread
    //Operation("group_barrier", {""}, 256, "", {OpType::Static}),  // TODO implement in graph

    //Allocation operations
    Operation("memory", {"_f", "_i", "_u", "_b"}, 0, "", {OpProp::Memory, OpProp::Special, OpProp::HostOnly, OpProp::Nondiff}),
    Operation("reshape", {"_f", "_i", "_u", "_b"}, 0, "", {OpProp::Memory, OpProp::Special, OpProp::HostOnly, OpProp::MemoryReuse}),
	Operation("assert", {"_f", "_i", "_u", "_b"}, 0, "assert_tensor", {OpProp::Memory, OpProp::Special, OpProp::HostOnly, OpProp::MemoryReuse}),
	Operation("input_shape", {"_i"}, 0, "", {OpProp::Special, OpProp::Static, OpProp::HostOnly, OpProp::Nondiff}),
    Operation("deallocate", {""}, 0, "", {OpProp::Memory, OpProp::Special, OpProp::HostOnly, OpProp::Nondiff}),
    //Operation("local_memory", {"_f", "_i", "_u"}, 0, "", {OpType::Memory, OpType::Special}), // TODO implement in graph

	Operation("region_begin", {""}, 0, "", {OpProp::Special, OpProp::Static, OpProp::HostOnly, OpProp::Nondiff, OpProp::Debug}),
	Operation("region_end", {""}, 0, "", {OpProp::Special, OpProp::Static, OpProp::HostOnly, OpProp::Nondiff, OpProp::Debug}),
	//Operation("print", {"f_", "i_", "u_", "b_"}, 0, "", {OpProp::Special, OpProp::Static, OpProp::HostOnly, OpProp::Nondiff, OpProp::Debug}),

    //Algorithms
    //Reduction
    Operation("dim_sum", {"f_f", "u_u", "i_i"}, 0, "", {OpProp::Algorithm, OpProp::Reduction}), // sum of the last dimension
    Operation("dim_norm", {"f_f", "u_u", "i_i"}, 0, "", {OpProp::Algorithm}), // length(norm) of the last dimension (cant just split dimensions so, didnt add reduction prop)
    Operation("dim_max", {"f_f", "u_u", "i_i"}, 0, "", {OpProp::Algorithm, OpProp::Reduction}), // max of the last dimension
    Operation("dim_min", {"f_f", "u_u", "i_i"}, 0, "", {OpProp::Algorithm, OpProp::Reduction}), // min of the last dimension
    Operation("dim_mean", {"f_f", "u_u", "i_i"}, 0, "", {OpProp::Algorithm, OpProp::Reduction}), // mean of the last dimension
    Operation("dim_prod", {"f_f", "u_u", "i_i"}, 0, "", {OpProp::Algorithm, OpProp::Reduction}), // product of the last dimension
    Operation("dim_any", {"u_u", "i_i", "b_b"}, 0, "", {OpProp::Algorithm, OpProp::Nondiff, OpProp::Reduction}), // any of the last dimension
    Operation("dim_all", {"u_u", "i_i", "b_b"}, 0, "", {OpProp::Algorithm, OpProp::Nondiff, OpProp::Reduction}), // all of the last dimension
	//Scan
	Operation("dim_prefix_sum", {"f_f", "u_u", "i_i"}, 0, "", {OpProp::Algorithm}),
    //Matrix
    Operation("transpose", {"f_f", "u_u", "i_i", "b_b"}, 0, "", {OpProp::Algorithm}),
    Operation("dot", {"ff_f"}, 0, "", {OpProp::Algorithm}), // dot product of the last dimensions
    Operation("matmul", {"ff_f"}, 0, "", {OpProp::Algorithm}), // matrix multiplication of the last dimensions
 	//Texture
	//Operation("interp1d", {"f_f"}, 0, "", {OpProperty::Algorithm}),
	//Operation("interp2d", {"f_f"}, 0, "", {OpProperty::Algorithm}),
	//Operation("interp3d", {"f_f"}, 0, "", {OpProperty::Algorithm}),
	//Other
	Operation("dim_reverse", {"f_f", "u_u", "i_i", "b_b"}, 0, "", {OpProp::Algorithm}),
	// Operation("dim_repeat", {"f_f", "u_u", "i_i", "b_b"}, 0, "", {OpProperty::Algorithm}),
	// Operation("tile_reduce", {"f_f", "u_u", "i_i", "b_b"}, 0, "", {OpProperty::Algorithm}),
	// Operation("dim_concat", {"ff_f", "uu_u", "ii_i", "bb_b"}, 0, "", {OpProperty::Algorithm}),
	Operation("dim_split", {"f_f", "u_u", "i_i", "b_b"}, 0, "", {OpProp::Algorithm}),
	// Operation("dim_pad", {"f_f", "u_u", "i_i", "b_b"}, 0, "", {OpProperty::Algorithm}),
	// Operation("split_dims", {"f_f", "u_u", "i_i", "b_b"}, 0, "", {OpProperty::Algorithm}),
	// Operation("merge_dims", {"f_f", "u_u", "i_i", "b_b"}, 0, "", {OpProperty::Algorithm}),
	Operation("unsqueeze", {"f_f", "u_u", "i_i", "b_b"}, 0, "", {OpProp::Algorithm}),
	Operation("squeeze", {"f_f", "u_u", "i_i", "b_b"}, 0, "", {OpProp::Algorithm}),

	//Native operations (built-in shader operations, only for size <= 4)
    //Operation("native_dot", {"ff_f"}, 0, "", {OpType::Static, OpType::Algorithm}),
    //Operation("native_matmul", {"ff_f"}, 0, "", {OpType::Static, OpType::Algorithm}),
    //Operation("native_norm", {"f_f"}, 0, "", {OpType::Static, OpType::Algorithm}),
    //Operation("native_max", {"f_f"}, 0, "", {OpType::Static, OpType::Algorithm}),
    //Operation("native_min", {"f_f"}, 0, "", {OpType::Static, OpType::Algorithm}),
    //Operation("native_any", {"u_u", "i_i", "b_b"}, 0, "", {OpType::Static, OpType::Algorithm}),
    //Operation("native_all", {"u_u", "i_i", "b_b"}, 0, "", {OpType::Static, OpType::Algorithm}),

    //Advanced
    //Operation("sort", {"_f", "_u", "_i"}, 0, "", {OpType::Static}),

    //Autodiff
    Operation("backwards_grad", {"ff_f"}, 0, "", {OpProp::Static, OpProp::Gradient}),
    Operation("forward_grad", {"ff_f"}, 0, "", {OpProp::Static, OpProp::Gradient}),

    // Memory operations
    //Operation("local_load", {"_f", "_u", "_i"}, 8, "", {OpType::Load}), // TODO implement in graph
    //Operation("local_store", {"f_", "u_", "i_"}, 8, "", {OpType::Store, OpType::Modifier}), // TODO implement in graph
    Operation("load", {"_f", "_u", "_i", "_b"}, 128, "",
              {OpProp::Load, OpProp::MemoryOp}),
    Operation("store", {"f_", "u_", "i_", "b_"}, 128, "",
              {OpProp::Store, OpProp::MemoryOp, OpProp::Modifier}),
    Operation("set", {"f_", "u_", "i_", "b_"}, 1, "",
              {OpProp::Set, OpProp::Modifier}),
    Operation("InterlockedAdd", {"u_", "i_", "f_"}, 256, "",
              {OpProp::Scatter, OpProp::MemoryOp, OpProp::Modifier}),
    Operation("InterlockedMin", {"u_", "i_", "f_"}, 256, "",
              {OpProp::Scatter, OpProp::MemoryOp, OpProp::Modifier}),
    Operation("InterlockedMax", {"u_", "i_", "f_"}, 256, "",
              {OpProp::Scatter, OpProp::MemoryOp, OpProp::Modifier}),
    Operation("InterlockedAnd", {"u_", "i_"}, 256, "",
              {OpProp::Scatter, OpProp::MemoryOp, OpProp::Modifier, OpProp::Nondiff}),
    Operation("InterlockedOr", {"u_", "i_"}, 256, "",
              {OpProp::Scatter, OpProp::MemoryOp, OpProp::Modifier, OpProp::Nondiff}),
    Operation("InterlockedXor", {"u_", "i_"}, 256, "",
              {OpProp::Scatter, OpProp::MemoryOp, OpProp::Modifier, OpProp::Nondiff}),
    Operation("InterlockedAdd_Prev", {"u_u", "i_i", "f_f"}, 256, "",
              {OpProp::Scatter, OpProp::MemoryOp, OpProp::Modifier, OpProp::CantSubstitute, OpProp::Nondiff}),

    // Index operations
    Operation("dim_id", {"_i"}, 0, "dim", {OpProp::Nondiff}, OpClass::DimensionIndex),
    Operation("block_thread_id", {"_i"}, 0, "", {OpProp::Nondiff}, OpClass::DimensionIndex),
    Operation("block_id", {"_i"}, 0, "", {OpProp::Nondiff}, OpClass::Variable),

    //Compute operations
    Operation("copy", {"f_f", "u_u", "i_i", "b_b"}, 1, "", {}, OpClass::Copy), //TODO: make sure no one copies memory objects
    Operation("add", {"ff_f", "uu_u", "ii_i"}, 1, "+", {}, OpClass::Operator),
    Operation("sub", {"ff_f", "uu_u", "ii_i"}, 1, "-", {}, OpClass::Operator),
    Operation("mul", {"ff_f", "uu_u", "ii_i"}, 1, "*", {}, OpClass::Operator),
    Operation("div", {"ff_f", "uu_u", "ii_i"}, 2, "/", {}, OpClass::Operator),
    Operation("mod", {"ff_f", "uu_u", "ii_i"}, 4, "%", {OpProp::Nondiff}, OpClass::Operator),
    Operation("lshift", {"uu_u", "ui_u", "ii_i"}, 1, "<<", {OpProp::Nondiff}, OpClass::Operator),
    Operation("rshift", {"uu_u", "ui_u", "ii_i"}, 1, ">>", {OpProp::Nondiff}, OpClass::Operator),
    Operation("and", {"uu_u", "ii_i", "bb_b"}, 1, "&", {OpProp::Nondiff}, OpClass::Operator),
    Operation("or", {"uu_u", "ii_i", "bb_b"}, 1, "|", {OpProp::Nondiff}, OpClass::Operator),
    Operation("xor", {"uu_u", "ii_i", "bb_b"}, 1, "^", {OpProp::Nondiff}, OpClass::Operator),
    Operation("eq", {"ff_b", "uu_b", "ii_b", "bb_b"}, 1, "==", {OpProp::Nondiff}, OpClass::Operator),
    Operation("neq", {"ff_b", "uu_b", "ii_b", "bb_b"}, 1, "!=", {OpProp::Nondiff}, OpClass::Operator),
    Operation("lt", {"ff_b", "uu_b", "ii_b", "bb_b"}, 1, "<", {OpProp::Nondiff}, OpClass::Operator),
    Operation("lte", {"ff_b", "uu_b", "ii_b", "bb_b"}, 1, "<=", { OpProp::Nondiff}, OpClass::Operator),
    Operation("gt", {"ff_b", "uu_b", "ii_b", "bb_b"}, 1, ">", {OpProp::Nondiff}, OpClass::Operator),
    Operation("gte", {"ff_b", "uu_b", "ii_b", "bb_b"}, 1, ">=", {OpProp::Nondiff}, OpClass::Operator),
    Operation("not", {"b_b", "u_u", "i_i"}, 1, "!", {OpProp::Nondiff}, OpClass::UnaryOperator),
    Operation("neg", {"f_f", "u_u", "i_i"}, 1, "-", {}, OpClass::UnaryOperator),
    Operation("uint", {"f_u", "u_u", "i_u", "b_u"}, 1, "uint", {OpProp::Nondiff}, OpClass::TypeCast),
    Operation("int", {"f_i", "u_i", "i_i", "b_i"}, 1, "int", {OpProp::Nondiff}, OpClass::TypeCast),
    Operation("float", {"f_f", "u_f", "i_f", "b_f"}, 1, "float", {OpProp::Nondiff}, OpClass::TypeCast),
    Operation("bool", {"f_b", "u_b", "i_b", "b_b"}, 1, "bool", {OpProp::Nondiff}, OpClass::TypeCast),
    Operation("asuint", {"f_u", "u_u", "i_u", "b_u"}, 0, "asuint",
              {OpProp::Nondiff}, OpClass::TypeReinterpret),
    Operation("asint", {"f_i", "u_i", "i_i", "b_i"}, 0, "asint",
              {OpProp::Nondiff}, OpClass::TypeReinterpret),
    Operation("asfloat", {"f_f", "u_f", "i_f", "b_f"}, 0, "asfloat",
              {OpProp::Nondiff}, OpClass::TypeReinterpret),
	Operation("asbool", {"f_b", "u_b", "i_b", "b_b"}, 0, "asbool",
			  {OpProp::Nondiff}, OpClass::TypeReinterpret),
    Operation("min", {"ff_f", "uu_u", "ii_i"}, 1),
    Operation("max", {"ff_f", "uu_u", "ii_i"}, 1),
    Operation("abs", {"f_f", "u_u", "i_i"}, 1),
    Operation("sign", {"f_f", "i_i"}, 1),
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
	Operation("reversebits", {"u_u"}, 8),
    Operation("pcgf", {"u_f"}, 34),
    Operation("pow", {"ff_f"}, 6),
    Operation("atan2", {"ff_f"}, 32),
    Operation("modf", {"ff_f"}, 2),
    Operation("step", {"ff_f"}, 2),
    Operation("clamp", {"fff_f", "uuu_u", "iii_i"}, 4),
    Operation("lerp", {"fff_f"}, 4),
    Operation("fma", {"fff_f"}, 1),
    Operation("smoothstep", {"fff_f"}, 10),
    Operation("ternary", {"bff_f", "buu_u", "bii_i", "bbb_b"}, 4, "", {}, OpClass::TernaryOperator),
    Operation("const", {"_f", "_u", "_i", "_b"}, 0, "", {OpProp::Nondiff}, OpClass::Constant),
};

unordered_map<string, const Operation*> CreateOperationMap() {
    unordered_map<string, const Operation*> operation_map;
    for (const auto& op : operations) {
        if (operation_map.contains(op.name_)) {
			throw runtime_error("Operation already exists: " + op.name_);
		}
		operation_map[op.name_] = &op;
	}
    return operation_map;
}

unordered_map<string, const Operation*> operation_map = CreateOperationMap();

DataTypeList Types(initializer_list<TFType> elements) {
	return DataTypeList(elements);
}

const Operation* FindOperation(const string& name) {
	if (name == "") {
		throw runtime_error("Operation name is empty");
	}

    auto it = operation_map.find(name);
    if (it != operation_map.end()) {
		return it->second;
	}

	throw runtime_error("IR Operation not defined: " + name);
}

string DataTypeToString(TFType type) { return type_names[type]; }

string RemoveSpaces(string str) {
	str.erase(remove(str.begin(), str.end(), ' '), str.end());
	return str;
}

}  // namespace TensorFrost
