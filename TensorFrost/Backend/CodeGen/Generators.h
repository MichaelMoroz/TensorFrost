#pragma once
#include <regex>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <string>
#include "Compiler/KernelGen.h"
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

string GetNodeString(const Node* node, bool verbose = false);
string GetOperationListing(const IR&, bool compact = false,
						   map<Node*, string> invalid = {});

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

	CodeGenerator(IR* ir) : ir(ir) {}

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
		for (auto& arg : args.Inputs()) {
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
				bool has_single_output = (node->args.Outputs().size() == 1) || is_constant || is_variable;
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
			debug = "v" + node->name;//return;
		}
		if (IsForbiddenName(debug)) {
			debug = debug + "0";
		}
		// check if the name is already used
		if (name_count.contains(debug)) {
			name_count[debug]++;
			debug = debug + "_" + to_string(name_count[debug]);
		} else {
			name_count[debug] = 1;
		}

		node->var_name = debug;
	}

	Line* GenerateLine(Node* node);

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