#pragma once
#include <regex>
#include <sstream>
#include <string>
#include <format>
#include "IR/KernelGen.h"
#include "Tensor/Tensor.h"

namespace TensorFrost {

string GetNodeName(const Node* node,  bool compact = false);
void GenerateNodeNames(const IR& ir);

string GenerateCPP(Program* program);

void GenerateCode(Program* program);

using ArgumentNames = map<ArgID, string>;

class Line {
 public:
	string left;
	string expression;
	string right;
	string name;
	int indent;
	float cost = 0;

	Line(string left, string expression, string right, string name, float cost = 0, int indent = 0)
	    : left(left), right(right), name(name), indent(indent), expression(expression), cost(cost) {}

	Line(int indent, string expression)
	    : indent(indent), expression(expression), cost(0), left(""), right(""), name("") {}
};

class CodeGenerator {
 public:
	list<Line*> lines;
	map<Node*, string> custom_generated_code_;

	CodeGenerator() = default;

	virtual Line* GenerateLine(Node* node, map<Node*, int> offsets, map<Node*, int> variables) = 0;
	virtual void GenerateArgumentNames(ArgumentManager& args, map<Node*, int> variables) = 0;

	void GenerateKernelLines(const IR* ir, const Node* cluster,
	                         const Kernel* kernel);
	void Compactify();
	string GetFinalCode();

protected:

	//	// generate line
	//string left = "";
	//string expression = "";
	//string right = "";
	//bool needs_parenthesis = true;
	//if (op->name_ == "loop") {
	//	 left += "for (int " + name + " = " + args.Name(ArgType::Input, 0) + "; " +
	//			   name + " < " + args.Name(ArgType::Input, 1) + "; " + name +
	//			   " += " + args.Name(ArgType::Input, 2) + ")";
	//} else if (op->name_ == "if") {
	//	 left += "if (" + args.Name(ArgType::Input, 0) + ")";
	//} else if (op->HasAllTypes(OpType::MemoryOp)) {
	//	 string address;
	//	 if (offset_array) {
	//		 address = offset_name_ + "[" +
	//					 to_string(offsets[args.Get(ArgType::Memory)]) + "]";
	//	 } else {
	//		 address = args.Name(ArgType::Memory);
	//	 }
	//
	//	 // if has index (not a scalar)
	//	 if (args.Has(ArgType::Index)) {
	//		 address += " + " + args.Name(ArgType::Index);
	//	 }
	//	 string memory_expression = "mem[" + address + "]";
	//	 if (op->name_ == "load") {
	//		 left += type_names[output_type] + " " + name + " = ";
	//		 if (output_type == DataType::Float) {
	//			 expression += "asfloat(";
	//		 }
	//		 if (output_type == DataType::Int) {
	//			 expression += "asint(";
	//		 }
	//		 expression += memory_expression;
	//		 if (output_type != DataType::Uint) {
	//			 expression += ")";
	//		 }
	//		 right += ";";
	//		 needs_parenthesis = false;
	//	 } else if (op->name_ == "store") {
	//		 expression += memory_expression + " = ";
	//		 if (args.Type(ArgType::Memory) != DataType::Uint) {
	//			 expression += "asuint(";
	//		 }
	//		 expression += args.Name(ArgType::Input, 0);
	//		 if (args.Type(ArgType::Memory) != DataType::Uint) {
	//			 expression += ")";
	//		 }
	//		 right += ";";
	//	 } else if (op->HasAllTypes(OpType::Scatter)) {
	//		 if (output_type != DataType::None) {
	//			 left += type_names[output_type] + " " + name + " = ";
	//		 }
	//		 string input_type_name = type_names[args.Type(ArgType::Input)];
	//		 expression += op->code_ + "((" + input_type_name + "*)mem" + ", " +
	//						 address + ", " + args.Name(ArgType::Input) + ")";
	//		 right += ";";
	//	 }
	//} else if (op->name_ == "set") {
	//	 left += args.Name(ArgType::Memory) + " = ";
	//	 expression += args.Name(ArgType::Input);
	//	 right += ";";
	//} else if (op->name_ == "memory") {
	//	 left += "uint " + node->var_name + " = ";
	//	 // if input memory type then just take the input and store it in the
	//	 // output
	//	 if (node->memory_type_ == MemoryType::Input ||
	//		   node->memory_type_ == MemoryType::Shape) {
	//		 expression += "in[" + to_string((*input_memory_index)++) + "]";
	//		 right += ";";
	//	 }
	//	 // if any other memory type - allocate it
	//	 else {
	//		 // get shape arguments
	//		 ArgMap args = node->GetArgumentMap(ArgType::Shape);
	//		 int dims = (int)args.size();
	//
	//		 string shape_arg = "{";
	//		 if (dims == 0) {
	//			 shape_arg += "1";
	//		 } else {
	//			 for (int j = 0; j < dims; j++) {
	//				 if (j != 0) {
	//					 shape_arg += ", ";
	//				 }
	//				 Node* shape_node = args[j]->from_->get();
	//
	//				 shape_arg += "(uint)" + ReadVariable(shape_node);
	//			 }
	//		 }
	//
	//		 shape_arg += "}";
	//
	//		 expression += "allocate(alloc, mem, " + shape_arg + ")";
	//		 right += ";";
	//	 }
	//} else if (op->name_ == "deallocate") {
	//	 left = "deallocate(" + args.Name(ArgType::Memory) + ")";
	//	 right = ";";
	//} else {
	//	 if (output_type != DataType::None) {
	//		 left += type_names[output_type] + " " + name + " = ";
	//	 }
	//	 string line;
	//
	//	 switch (op->op_types_[0]) {  // TODO: properly support multiple op types
	//		 case OpType::Operator:
	//			 line += args.Name(ArgType::Input, 0) + " " + op->code_ + " " +
	//					   args.Name(ArgType::Input, 1);
	//			 break;
	//		 case OpType::UnaryOperator:
	//			 line += op->code_ + args.Name(ArgType::Input, 0);
	//			 break;
	//		 case OpType::Function:
	//			 line += op->code_ + "(";
	//			 for (int i = 0; i < args.Count(ArgType::Input); i++) {
	//				 if (i != 0) {
	//					 line += ", ";
	//				 }
	//				 line += args.Name(ArgType::Input, i);
	//			 }
	//			 line += ")";
	//			 needs_parenthesis = false;
	//			 break;
	//		 case OpType::Keyword:
	//			 line += op->code_;
	//			 break;
	//		 case OpType::DimensionIndex:
	//			 line += op->code_ + to_string(node->GetTensor()->data[0]);
	//			 needs_parenthesis = false;
	//			 break;
	//		 case OpType::TypeCast:
	//			 line += "(" + op->code_ + ")" + args.Name(ArgType::Input, 0);
	//			 break;
	//		 case OpType::TypeReinterpret:
	//			 line += "*(" + op->code_ + "*)&" + args.Name(ArgType::Input, 0);
	//			 break;
	//		 case OpType::Constant:
	//			 line += node->GetTensor()->GetConstantString();
	//			 needs_parenthesis = false;
	//			 break;
	//		 case OpType::TernaryOperator:
	//			 line += args.Name(ArgType::Input, 0) + " ? " +
	//					   args.Name(ArgType::Input, 1) + " : " +
	//					   args.Name(ArgType::Input, 2);
	//			 break;
	//		 default:
	//			 line += "";
	//			 break;
	//	 }
	//	 expression += line;
	//	 right += ";";
	//}

	string GenerateLoop(ArgumentManager* args, const string& name)
	{
		string in1 = args->Name(ArgType::Input, 0), in2 = args->Name(ArgType::Input, 1), in3 = args->Name(ArgType::Input, 2);
		return "for (int " + name + " = " + in1 + "; " + name + " < " + in2 + "; " + name + " += " + in3 + ")";
	}

	string GenerateIf(ArgumentManager* args)
	{
		return "if (" + args->Name(ArgType::Input, 0) + ")";
	}

	string GenerateMemoryOp(ArgumentManager* args, const string& name, const string& memory_expression, const string& type_name, const string& output_type, bool needs_parenthesis)
	{
		string left, expression, right;
		left = type_name + " " + name + " = ";
		if (output_type == "float") {
			expression += "asfloat(";
		}
		if (output_type == "int") {
			expression += "asint(";
		}
		expression += memory_expression;
		if (output_type != "uint") {
			expression += ")";
		}
		right += ";";
		needs_parenthesis = false;
		return left + expression + right;
	}



};

string AddIndent(const string& input, const string& indent);


}  // namespace TensorFrost