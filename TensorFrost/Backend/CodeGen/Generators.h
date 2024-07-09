#pragma once
#include <regex>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <string>
#include "IR/KernelGen.h"
#include "Tensor/Tensor.h"
#include "Backend/Backend.h"

namespace TensorFrost {

string GetNodeName(const Node* node,  bool compact = false);
string ReadVariable(Node* node);
void GenerateNodeNames(const IR& ir);

string GetBufferDeclarations(Kernel* kernel, function<string(const string&, const string&, size_t)> get_name);
string GetCPPHeader();
string GetCPPImplementation();
string GetHLSLHeader(Kernel* kernel);
string GetGLSLHeader(Kernel* kernel);
void GenerateMain(Program* program, map<Node*, string>& dispatch_code, int input_count, int output_count);
void GenerateKernel(Program* program, Kernel* kernel);
void GenerateCPPKernel(Program* program, Kernel* kernel);
void GenerateHLSLKernel(Program* program, Kernel* kernel);
void GenerateGLSLKernel(Program* program, Kernel* kernel);
void GenerateCode(Program* program);

bool IsForbiddenName(const string& name);

using ArgumentNames = map<ArgID, string>;

class Line {
 public:
	Node* node;
	string left;
	string expression;
	string right;
	string name;
	int indent;

	Line(Node* node, string left, string expression, string right, string name, int indent = 0)
	    : left(left), right(right), name(name), indent(indent), expression(expression), node(node) {}

	Line(int indent, string expression)
	    : indent(indent), expression(expression), left(""), right(""), name(""), node(nullptr) {}
};

class CodeGenerator {
protected:
	unordered_map<string, string> name_map_;
 public:
	list<Line*> lines;
	map<Node*, string> custom_generated_code_;
	
	Kernel* kernel = nullptr;
	IR* ir = nullptr;

	CodeGenerator(IR* ir) : ir(ir) {
		name_map_ = {
			{"var", "var_"},
		};
	}

	void GenerateKernelCode(Kernel *kernel_);
	void GenerateCode(const Node* root);
	string AssembleString();

protected:
	map<Node*, size_t> read_write_bindings;
	map<Node*, size_t> read_only_bindings;
	map<Node*, size_t> variables;
	map<Node*, string> node_expression;
	map<Node*, bool> requires_paranthesis;
	unordered_set<Node*> lines_to_remove;
	vector<string> additional_lines;
	unordered_map<string, int> name_count;

	virtual void GenerateArgumentNames(ArgumentManager& args)  {
		for (auto& arg : args.inputs_) {
			Node* node = arg.second;
			ArgID id = arg.first;
			string name = node->var_name;
			bool need_parenthesis = false;
			if (variables.contains(node)) {
				name = GetName("var") + name;
			} else {
				string expr = node_expression[node];
				bool is_memory = node->op->HasAllTypes(OpProp::Memory);
				bool is_static = node->op->HasAllTypes(OpProp::Static) ||
								 node->op->HasAllTypes(OpProp::CantSubstitute);
				bool is_constant = node->op->class_ == OpClass::Constant;
				bool is_variable = node->op->class_ == OpClass::Variable;
				if (is_constant && expr == "") {
					expr = node->GetTensor()->GetConstantString();
				}
				bool has_name = node->debug_name != "";
				bool has_single_output = (node->args.outputs_.size() == 1) || is_constant || is_variable;
				bool modified = node->flags.has(NodeProp::Modified);
				bool short_enough = expr.size() < 100;
				bool can_substitude = !has_name && has_single_output && !modified && short_enough && !is_static && !is_memory;
				if (can_substitude) {
					if (expr == "") {
						throw std::runtime_error("Substitute expression is empty");
					}
					name = expr;
					need_parenthesis = requires_paranthesis[node];
					lines_to_remove.insert(node);
				}
			}
			args.SetName(id, name, need_parenthesis);
		}
	}

	void RegenerateNodeName(Node* node) {
		string debug = node->debug_name;
		if (debug.empty()) {
			return;
		}
		// check if the name is already used
		if (name_count.contains(debug)) {
			name_count[debug]++;
			debug = debug + "_" + to_string(name_count[debug]);
		} else {
			name_count[debug] = 1;
		}
		if (IsForbiddenName(debug)) {
			debug = debug + "0";
		}
		node->var_name = debug;
	}

	virtual Line* GenerateLine(Node* node)  {
		ArgumentManager& args = node->args;
		GenerateArgumentNames(args);
		if (kernel) RegenerateNodeName(node);
		const Operation* op = node->op;

		string name = node->var_name;

		// get output type
		TFType output_type = node->type;

		// generate line
		string left = "";
		string expression = "";
		string right = "";
		bool needs_paranthesis = false;

		if (op->HasAllTypes(OpProp::Special)) {
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

					shape_arg += "(uint)" + args.Name(ArgType::Shape, j);
				}
			}

			shape_arg += "}";

			if (op->name_ == "loop") {
				left += GenerateLoop(&args, name);
			} else if (op->name_ == "if") {
				left += GenerateIf(&args);
			} else if (op->name_ == "memory") {
				// if input memory type then just take the input and store it in the
				// output
				if (node->flags.has(NodeProp::InputMemory)) {
					left += "tf.check_tensor(" + node->var_name+ ", \"" + node->var_name + "\", " + shape_arg + ", TFType::" + DataTypeNames[output_type] + ")";
					right += ";";
				}
				// if any other memory type - allocate it
				else {
					left += "TFTensor " + node->var_name + " = ";
					expression += "tf.allocate(\"" + node->var_name + "\", " + shape_arg + ", TFType::" + DataTypeNames[output_type] + ")";
					right += ";";
				}
			} else if (op->name_ == "deallocate") {
				left = "tf.deallocate(" + args.Name(ArgType::Memory) + ")";
				right = ";";
			}
			else if (op->name_ == "input_shape")
			{
				left = "int " + node->var_name + " = ";
				expression = ir->input_memory_map[node->flags.get(NodeProp::InputShapeMemory)]->var_name + ".shape[" + to_string(node->flags.get(NodeProp::InputShapeDim)) + "]";
				right = ";";
			}
			else if (op->name_ == "reshape")
			{
				left = "TFTensor " + node->var_name + " = ";
				expression = "tf.reshape(" + args.Name(ArgType::Memory) + ", \"" + node->var_name + "\", " + shape_arg + ", TFType::" + DataTypeNames[output_type] + ")";
				right = ";";
			}
		} else if (op->HasAllTypes(OpProp::MemoryOp)) {
			string address;

			if (kernel) {
				address = "0";
				// if has index (not a scalar)
				if (args.Has(ArgType::Index)) {
					address = args.Name(ArgType::Index);
				}

				string memory_expression = args.Name(ArgType::Memory) + "_mem[" + address + "]";
				if (op->name_ == "load") {
					string output_type_name = type_names[output_type];
					left += output_type_name + " " + name + " = ";
					expression +=
					    (output_type == Uint)
					        ? memory_expression
					        : TypeReinterpret(output_type_name, memory_expression);
					right += ";";
				} else if (op->name_ == "store") {
					expression += memory_expression + " = ";
					expression +=
					    (output_type == Uint)
					        ? args.Name(ArgType::Input)
					        : TypeReinterpret("uint", args.Name(ArgType::Input));
					right += ";";
				} else if (op->HasAllTypes(OpProp::Scatter)) {
					if (output_type != None) {
						left += type_names[output_type] + " " + name + " = ";
					}
					string output_type_name = type_names[output_type];
					string input_type_name = type_names[args.Type(ArgType::Input)];
					expression += GenerateAtomicOp(op->name_, input_type_name,
					                               output_type_name, address,
					                               args.Name(ArgType::Input), name, args.Name(ArgType::Memory));
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
					string memory_expression = GetName("tf.read") + "(" + tensor_name + ", " + address + ")";
					expression += (output_type == Uint)
						? memory_expression
						: TypeReinterpret(output_type_name, memory_expression);
					right += ";";
				} else if (op->name_ == "store") {
					//do writeback
					string memory_expression = GetName("tf.write") + "(" + tensor_name + ", " + address + ", ";
					expression += memory_expression + args.Name(ArgType::Input) + ")";
					right += ";";
				} else if (op->HasAllTypes(OpProp::Scatter)) {
					throw std::runtime_error("Scatter operation not supported in non-kernel mode");
				}
			}
			
		} else if (op->name_ == "set") {
			left += args.Name(ArgType::Memory) + " = ";
			expression += args.Name(ArgType::Input);
			right += ";";
		} else {
			if (output_type != None) {
				left += type_names[output_type] + " " + name + " = ";
			}
			string line;
			string code = op->code_;
			switch (op->class_) {
				case OpClass::Operator:
					args.AddParenthesis(true);
					if ((code == "&" || code == "|") && output_type == Bool) {
						code = code + code;
					}
					line += args.Name(ArgType::Input, 0) + " " + code + " " +
					        args.Name(ArgType::Input, 1);
					needs_paranthesis = true;
					break;
				case OpClass::UnaryOperator:
					args.AddParenthesis(true);
					line += op->code_ + args.Name(ArgType::Input, 0);
					needs_paranthesis = true;
					break;
				case OpClass::Function:
					line += GetName(op->code_) + "(";
					for (int i = 0; i < args.Count(ArgType::Input); i++) {
						if (i != 0) {
							line += ", ";
						}
						line += args.Name(ArgType::Input, i);
					}
					line += ")";
					break;
				case OpClass::Copy:
					line += args.Name(ArgType::Input, 0);
				    needs_paranthesis = true;
					break;
				case OpClass::Keyword:
					line += op->code_;
					break;
				case OpClass::DimensionIndex:
					line += op->code_ + to_string(node->data[0]);
					break;
				case OpClass::Variable:
					line += op->code_;
					break;
				case OpClass::TypeCast:
					line += GenerateTypeCast(&args, op->code_);
					break;
				case OpClass::TypeReinterpret:
					line += GenerateTypeReinterpret(&args, op->code_);
					break;
				case OpClass::Constant:
					line += node->GetTensor()->GetConstantString();
					break;
				case OpClass::TernaryOperator:
					args.AddParenthesis(true);
					line += args.Name(ArgType::Input, 0) + " ? " +
					        args.Name(ArgType::Input, 1) + " : " +
					        args.Name(ArgType::Input, 2);
					needs_paranthesis = true;
					break;
				default:
					throw std::runtime_error("Unknown operation class");
					break;
			}
			expression += line;
			right += ";";
		}

		node_expression[node] = expression;
		requires_paranthesis[node] = needs_paranthesis;

		return new Line(node, left, expression, right, name);
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
		return "((" + type_name + ")(" + input + "))";
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

	virtual string GenerateAtomicOp(const string& op,
	                                const string& input_type_name,
	                                const string& output_type_name,
	                                const string& address, const string& input, const string& output, const string& memory_name)
	{
		return op + "((" + input_type_name + "*)"+memory_name+"_mem" + ", " + address + ", " + input + ")";
	}

	string GetName(const string& name) {
		// Check if the function name is in the map
		if (name_map_.find(name) != name_map_.end()) {
			return name_map_[name];
		}

		// If not, return the original name
		return name;
	}
};

string AddIndent(const string& input, const string& indent);


}  // namespace TensorFrost