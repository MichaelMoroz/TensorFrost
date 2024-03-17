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

string ReadVariable(Node* node);
string GenerateCPP(Program* program);
string GenerateHLSLKernel(Program* program, const Kernel* kernel);
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

	string offset_name_ = "off";
	string variable_name_ = "var";

	map<DataType, string> type_names = {
	    {DataType::None, "void"},   {DataType::Bool, "bool"},
	    {DataType::Float, "float"}, {DataType::Uint, "uint"},
	    {DataType::Int, "int"},
	};
	
	bool offset_array = true;
	int* input_memory_index = nullptr;

	CodeGenerator() = default;

	void GenerateKernelCode(const Kernel* kernel);
	void GenerateCode(const Node* root);
	string AssembleString();

protected:
	map<Node*, int> offsets;
	map<Node*, int> variables;

	virtual void GenerateArgumentNames(ArgumentManager& args)  {
		for (auto& arg : args.arguments_) {
			string name = GetNodeName(arg.second, true);
			if (variables.contains(arg.second)) {
				name = variable_name_ + "[" + to_string(variables[arg.second]) + "]";
				name =
				    "as" + type_names[arg.second->GetTensor()->type] + "(" + name + ")";
			}
			args.SetName(arg.first, name);
		}
	}

	virtual Line* GenerateLine(Node* node)  {
		// TODO: Create argument manager class
		ArgumentManager args = node->GetArgumentManager();
		GenerateArgumentNames(args);
		const Operation* op = node->op;
		string name = node->var_name;

		// get output type
		DataType output_type = node->tensor_->type;

		// generate line
		string left = "";
		string expression = "";
		string right = "";

		if (op->HasAllTypes(OpType::Special)) {
			if (op->name_ == "loop") {
				left += GenerateLoop(&args, name);
			} else if (op->name_ == "if") {
				left += GenerateIf(&args);
			} else if (op->name_ == "memory") {
				left += "uint " + node->var_name + " = ";
				// if input memory type then just take the input and store it in the
				// output
				if (node->memory_type_ == MemoryType::Input ||
				    node->memory_type_ == MemoryType::Shape) {
					expression += "in[" + to_string((*input_memory_index)++) + "]";
					right += ";";
				}
				// if any other memory type - allocate it
				else {
					// get shape arguments
					ArgMap args = node->GetArgumentMap(ArgType::Shape);
					int dims = (int)args.size();

					string shape_arg = "{";
					if (dims == 0) {
						shape_arg += "1";
					} else {
						for (int j = 0; j < dims; j++) {
							if (j != 0) {
								shape_arg += ", ";
							}
							Node* shape_node = args[j]->from_->get();

							shape_arg += "(uint)" + ReadVariable(shape_node);
						}
					}

					shape_arg += "}";

					expression += "allocate(alloc, mem, " + shape_arg + ")";
					right += ";";
				}
			} else if (op->name_ == "deallocate") {
				left = "deallocate(" + args.Name(ArgType::Memory) + ")";
				right = ";";
			}
		} else if (op->HasAllTypes(OpType::MemoryOp)) {
			string address;

			if (offset_array) {
				address = offset_name_ + "[" +
				          to_string(offsets[args.Get(ArgType::Memory)]) + "]";
			} else {
				address = args.Name(ArgType::Memory);
			}
			// if has index (not a scalar)
			if (args.Has(ArgType::Index)) {
				address += " + " + args.Name(ArgType::Index);
			}

			string memory_expression = "mem[" + address + "]";
			if (op->name_ == "load") {
				string output_type_name = type_names[output_type];
				left += output_type_name + " " + name + " = ";
				expression += (output_type == DataType::Uint) ? memory_expression : TypeReinterpret(output_type_name, memory_expression);
				right += ";";
			} else if (op->name_ == "store") {
				expression += memory_expression + " = ";
				expression += (output_type != DataType::Uint) ? args.Name(ArgType::Input) : TypeReinterpret("uint", args.Name(ArgType::Input));
				right += ";";
			} else if (op->HasAllTypes(OpType::Scatter)) {
				if (output_type != DataType::None) {
					left += type_names[output_type] + " " + name + " = ";
				}
				string input_type_name = type_names[args.Type(ArgType::Input)];
				expression += op->code_ + "((" + input_type_name + "*)mem" + ", " +
				              address + ", " + args.Name(ArgType::Input) + ")";
				right += ";";
			}
		} else if (op->name_ == "set") {
			left += args.Name(ArgType::Memory) + " = ";
			expression += args.Name(ArgType::Input);
			right += ";";
		} else {
			if (output_type != DataType::None) {
				left += type_names[output_type] + " " + name + " = ";
			}
			string line;

			switch (op->op_types_[0]) {
				case OpType::Operator:
					line += args.Name(ArgType::Input, 0) + " " + op->code_ + " " +
					        args.Name(ArgType::Input, 1);
					break;
				case OpType::UnaryOperator:
					line += op->code_ + args.Name(ArgType::Input, 0);
					break;
				case OpType::Function:
					line += op->code_ + "(";
					for (int i = 0; i < args.Count(ArgType::Input); i++) {
						if (i != 0) {
							line += ", ";
						}
						line += args.Name(ArgType::Input, i);
					}
					line += ")";
					break;
				case OpType::Keyword:
					line += op->code_;
					break;
				case OpType::DimensionIndex:
					line += op->code_ + to_string(node->GetTensor()->data[0]);
					break;
				case OpType::TypeCast:
					line += GenerateTypeCast(&args, op->code_);
					break;
				case OpType::TypeReinterpret:
					line += GenerateTypeReinterpret(&args, op->code_);
					break;
				case OpType::Constant:
					line += node->GetTensor()->GetConstantString();
					break;
				case OpType::TernaryOperator:
					line += args.Name(ArgType::Input, 0) + " ? " +
					        args.Name(ArgType::Input, 1) + " : " +
					        args.Name(ArgType::Input, 2);
					break;
				default:
					line += "";
					break;
			}
			expression += line;
			right += ";";
		}

		return new Line(left, expression, right, name, op->cost_);
	}

	string GenerateLoop(ArgumentManager* args, const string& name)
	{
		string in1 = args->Name(ArgType::Input, 0), in2 = args->Name(ArgType::Input, 1), in3 = args->Name(ArgType::Input, 2);
		return "for (int " + name + " = " + in1 + "; " + name + " < " + in2 + "; " + name + " += " + in3 + ")";
	}

	string GenerateIf(ArgumentManager* args)
	{
		return "if (" + args->Name(ArgType::Input, 0) + ")";
	}

	string TypeCast(string type_name, string input)
	{
		return "(" + type_name + ")" + input;
	}

	string TypeReinterpret(string type_name, string input)
	{
		return "*(" + type_name + "*)&" + input;
	}

	string GenerateTypeCast(ArgumentManager* args, const string& type_name)
	{
		return TypeCast(type_name, args->Name(ArgType::Input, 0));
	}

	string GenerateTypeReinterpret(ArgumentManager* args, const string& type_name)
	{
		return TypeReinterpret(type_name, args->Name(ArgType::Input, 0));
	}
};

string AddIndent(const string& input, const string& indent);


}  // namespace TensorFrost