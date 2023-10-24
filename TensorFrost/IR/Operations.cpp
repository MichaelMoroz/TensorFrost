#include "Operations.h"

namespace TensorFrost {
#define dtype(x) DataType::x

map<DataType, string> TypeNames = {
    {dtype(None), "void"},
    {dtype(Bool), "bool"},
    {dtype(Float), "float"},
    {dtype(Uint), "uint"},
    {dtype(Int), "int"},
};

vector<Operation> operations = {
    Operation("add", {"ff_f", "uu_u", "ii_i"}, "+", true),
    Operation("sub", {"ff_f", "uu_u", "ii_i"}, "-", true),
    Operation("mul", {"ff_f", "uu_u", "ii_i"}, "*", true),
    Operation("div", {"ff_f", "uu_u", "ii_i"}, "/", true),
    Operation("mod", {"ff_f", "uu_u", "ii_i"}, "%", true),
    Operation("lshift", {"uu_u", "ii_i"}, "<<", true),
    Operation("rshift", {"uu_u", "ii_i"}, ">>", true),
    Operation("and", {"uu_u", "ii_i"}, "&", true),
    Operation("or", {"uu_u", "ii_i"}, "|", true),
    Operation("xor", {"uu_u", "ii_i"}, "^", true),
    Operation("eq", {"ff_b", "uu_b", "ii_b"}, "==", true),
    Operation("neq", {"ff_b", "uu_b", "ii_b"}, "!=", true),
    Operation("lt", {"ff_b", "uu_b", "ii_b"}, "<", true),
    Operation("lte", {"ff_b", "uu_b", "ii_b"}, "<=", true),
    Operation("gt", {"ff_b", "uu_b", "ii_b"}, ">", true),
    Operation("gte", {"ff_b", "uu_b", "ii_b"}, ">=", true),
    Operation("uint", {"f_u", "u_u", "i_u"}),
    Operation("int", {"f_i", "u_i", "i_i"}),
    Operation("float", {"f_f", "u_f", "i_f"}),
    Operation("min", {"ff_f", "uu_u", "ii_i"}),
    Operation("max", {"ff_f", "uu_u", "ii_i"}),
    Operation("abs", {"f_f", "u_u", "i_i"}),
    Operation("sign", {"f_f", "u_u", "i_i"}),
    Operation("ceil", {"f_f"}),
    Operation("floor", {"f_f"}),
    Operation("round", {"f_f"}),
    Operation("frac", {"f_f"}),
    Operation("exp", {"f_f"}),
    Operation("exp2", {"f_f"}),
    Operation("log", {"f_f"}),
    Operation("log2", {"f_f"}),
    Operation("sqrt", {"f_f"}),
    Operation("rsqrt", {"f_f"}),
    Operation("rcp", {"f_f"}),
    Operation("sin", {"f_f"}),
    Operation("cos", {"f_f"}),
    Operation("tan", {"f_f"}),
    Operation("asin", {"f_f"}),
    Operation("acos", {"f_f"}),
    Operation("atan", {"f_f"}),
    Operation("sinh", {"f_f"}),
    Operation("cosh", {"f_f"}),
    Operation("tanh", {"f_f"}),
    Operation("pcg", {"u_u"}),
    Operation("pcgf", {"u_f"}),
    Operation("pow", {"ff_f"}),
    Operation("atan2", {"ff_f"}),
    Operation("mod", {"ff_f"}),
    Operation("step", {"ff_f"}),
    Operation("clamp", {"fff_f"}),
    Operation("lerp", {"fff_f"}),
    Operation("fma", {"fff_f"}),
    Operation("ternary", {"bff_f", "buu_u", "bii_i"}),
    Operation("load", {"f_f", "u_u", "i_i"}),
    Operation("store",{"ff_", "uu_", "ii_"}),
    Operation("const_memory", {"_f", "_i", "_u"}),
    Operation("input_memory", {"_f", "_i", "_u"}),
    Operation("InterlockedAdd", {"uuu_", "iiu_", "ffu_"}),
    Operation("InterlockedMin", {"uuu_", "iiu_", "ffu_"}),
    Operation("InterlockedMax", {"uuu_", "iiu_", "ffu_"}),
    Operation("InterlockedAnd", {"uuu_", "iiu_"}),
    Operation("InterlockedOr", {"uuu_", "iiu_"}),
    Operation("InterlockedXor", {"uuu_", "iiu_"}),
    Operation("variable", {"_f", "_u", "_i"}),
    Operation("const", {"_f", "_u", "_i"}),
    Operation("dim_id", {"_i", "_u"}),
    Operation("thread_id", {"_i", "_u"}),
    Operation("group_thread_id", {"_i", "_u"}),
    Operation("group_id", {"_i", "_u"}),
    Operation("group_count", {"_i", "_u"}),
    Operation("thread_count", {"_i", "_u"}),
    Operation("loop_begin", {"iii_i"}),
    Operation("loop_end", {"i_"}),
    Operation("if_begin", {"b_"}),
    Operation("if_end", {""}),
    Operation("break", {""}),
    Operation("continue", {""}),
    Operation("return", {""}),
    Operation("GroupMemoryBarrierWithGroupSync", {""}),
};

DataTypeList Types(initializer_list<DataType> elements) {
	return DataTypeList(elements);
}

const Operation& FindOperation(const string& name) {
	for (const auto& op : operations) {
        if (op.GetName() == name) {
            return op;
        }
    }
	throw runtime_error("Operation not found: " + name);
}

string DataTypeToString(DataType type) {
    return TypeNames[type];
}

string Operation::GenerateOpString(const vector<string>& arguments) const
{
    string line = "";
    if(is_operator_)
    {
        line += arguments[0] + " " + code_ + " " + arguments[1];
    }
    else
    {
        line += code_ + "(";
        for(int i = 0; i < arguments.size(); i++)
        {
            if(i != 0)
            {
                line += ", ";
            }
            line += arguments[i];
        }
        line += ")";
    }
    return line;
}

string Operation::GenerateLine(const string& var_name, const vector<string>& arguments, const vector<DataType>& input_types) const
{
    //get output type
    DataType output_type = GetOutputType(input_types);

    //generate line
    string line = TypeNames[output_type] + " " + var_name + " = ";

    //generate op string
    line += GenerateOpString(arguments);

    //add semicolon
    line += ";";

    return line;
}

}  // namespace TensorFrost
