#pragma once
#include <regex>
#include <sstream>
#include <string>
#include <format>
#include "IR/KernelGen.h"
#include "Tensor/Tensor.h"
#include "Backend/Backend.h"

namespace TensorFrost {

string GetNodeName(const Node* node,  bool compact = false);
string ReadVariable(Node* node);
void GenerateNodeNames(const IR& ir);

void GenerateMain(Program* program, map<Node*, string>& dispatch_code, int input_count, int output_count);
void GenerateKernel(Program* program, Kernel* kernel);
void GenerateCPPKernel(Program* program, Kernel* kernel);
void GenerateHLSLKernel(Program* program, Kernel* kernel);
void GenerateGLSLKernel(Program* program, Kernel* kernel);
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
	
	bool is_kernel = true;

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
			int dims = args.Count(ArgType::Shape);

			string shape_arg = "{";
			if (dims == 0) {
				shape_arg += "1";
			} else {
				for (int j = 0; j < dims; j++) {
					if (j != 0) {
						shape_arg += ", ";
					}
					Node* shape_node = args.Get(ArgType::Shape, j);

					shape_arg += "(uint)" + ReadVariable(shape_node);
				}
			}

			shape_arg += "}";

			if (op->name_ == "loop") {
				left += GenerateLoop(&args, name);
			} else if (op->name_ == "if") {
				left += GenerateIf(&args);
			} else if (op->name_ == "memory") {
				left += "TensorProp " + node->var_name + " = ";
				// if input memory type then just take the input and store it in the
				// output
				if (node->memory_type_ == MemoryType::Input) {
					expression += "check_tensor(in" + to_string(node->special_index_) + ", \"" + node->var_name + "\", " + shape_arg + ", DataType::" + DataTypeNames[output_type] + ")";
					right += ";";
				}
				// if any other memory type - allocate it
				else {
					expression += "allocate(" + shape_arg + ", DataType::" + DataTypeNames[output_type] + ")";
					right += ";";
				}
			} else if (op->name_ == "deallocate") {
				left = "deallocate(" + args.Name(ArgType::Memory) + ")";
				right = ";";
			}
			else if (op->name_ == "input_shape")
			{
				Node* output_memory = node->outputs_[0]->to_->get();
				left = "int " + node->var_name + " = ";
				expression = "in" + to_string(output_memory->special_index_) + ".shape[" + to_string(node->special_index_) + "]";
				right = ";";
			}
		} else if (op->HasAllTypes(OpType::MemoryOp)) {
			string address;

			if (is_kernel) {
				address = offset_name_ + "[" +
				          to_string(offsets[args.Get(ArgType::Memory)]) + "]";
				// if has index (not a scalar)
				if (args.Has(ArgType::Index)) {
					address += " + " + args.Name(ArgType::Index);
				}

				string memory_expression = "mem[" + address + "]";
				if (op->name_ == "load") {
					string output_type_name = type_names[output_type];
					left += output_type_name + " " + name + " = ";
					expression +=
					    (output_type == DataType::Uint)
					        ? memory_expression
					        : TypeReinterpret(output_type_name, memory_expression);
					right += ";";
				} else if (op->name_ == "store") {
					expression += memory_expression + " = ";
					expression +=
					    (output_type == DataType::Uint)
					        ? args.Name(ArgType::Input)
					        : TypeReinterpret("uint", args.Name(ArgType::Input));
					right += ";";
				} else if (op->HasAllTypes(OpType::Scatter)) {
					if (output_type != DataType::None) {
						left += type_names[output_type] + " " + name + " = ";
					}
					string input_type_name = type_names[args.Type(ArgType::Input)];
					expression += GenerateAtomicOp(op->name_, input_type_name, address,
					                               args.Name(ArgType::Input));
					right += ";";
				}
			} else {
				string tensor_name = args.Name(ArgType::Memory);
				string address = "0";
				if (args.Has(ArgType::Index)) {
					address = args.Name(ArgType::Index);
				}

				if (op->name_ == "load") {
					//do readback
					string output_type_name = type_names[output_type];
					left += output_type_name + " " + name + " = ";
					string memory_expression = "ReadFromMemory(" + tensor_name + ", " + address + ")";
					expression += (output_type == DataType::Uint)
						? memory_expression
						: TypeReinterpret(output_type_name, memory_expression);
					right += ";";
				} else if (op->name_ == "store") {
					//do writeback
					string memory_expression = "WriteToMemory(" + tensor_name + ", " + address + ", ";
					expression += memory_expression + args.Name(ArgType::Input) + ")";
					right += ";";
				} else if (op->HasAllTypes(OpType::Scatter)) {
					throw std::runtime_error("Scatter operation not supported in non-kernel mode");
				}
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
				case OpType::Variable:
					line += op->code_;
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

	virtual string GenerateLoop(ArgumentManager* args, const string& name)
	{
		string in1 = args->Name(ArgType::Input, 0), in2 = args->Name(ArgType::Input, 1), in3 = args->Name(ArgType::Input, 2);
		return "for (int " + name + " = " + in1 + "; " + name + " < " + in2 + "; " + name + " += " + in3 + ")";
	}

	virtual string GenerateIf(ArgumentManager* args)
	{
		return "if (" + args->Name(ArgType::Input, 0) + ")";
	}

	virtual string TypeCast(string type_name, string input)
	{
		return "(" + type_name + ")" + input;
	}

	virtual string TypeReinterpret(string type_name, string input) {
		return "as" + type_name + "(" + input + ")";
	}

	virtual string GenerateTypeCast(ArgumentManager* args, const string& type_name)
	{
		return TypeCast(type_name, args->Name(ArgType::Input, 0));
	}

	virtual string GenerateTypeReinterpret(ArgumentManager* args, const string& type_name)
	{
		return TypeReinterpret(type_name, args->Name(ArgType::Input, 0));
	}

	virtual string GenerateAtomicOp(const string& op, const string& input_type_name, const string& address, const string& input)
	{
		return op + "((" + input_type_name + "*)mem" + ", " + address + ", " + input + ")";
	}
};

string AddIndent(const string& input, const string& indent);


}  // namespace TensorFrost