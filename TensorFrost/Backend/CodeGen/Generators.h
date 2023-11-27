#pragma once
#include <regex>
#include "IR/KernelGen.h"
#include "Tensor/Tensor.h"

namespace TensorFrost {
using NodeNames = std::unordered_map<const Node*, string>;

NodeNames GenerateNodeNames(const IR& ir);
string GetNodeName(const Node* node, NodeNames& names, bool compact = false);
string GetOperationListing(const IR&, bool compact = false);

string GenerateHLSL(const IR&);
string GenerateKernelHLSL(const IR&, const Lable*);

pair<string, vector<string>> GenerateC(Program* program);


class Line {
 public:
	string left;
	string expression;
	string right;
	string name;
	vector<string> arguments;
	int indent;
	bool needs_parenthesis = false;
	float cost = 0;

	Line(string left, string expression, string right, string name,
	     vector<string> args, bool needs_parenthesis = false, float cost = 0, int indent = 0)
	    : left(left), right(right), name(name), arguments(args), indent(indent), expression(expression), needs_parenthesis(needs_parenthesis), cost(cost) {}
};

class CodeGenerator {
 public:
	list<Line*> lines;

	CodeGenerator() = default;

	virtual Line* GenerateLine(NodeNames* names, const Operation* op, Node* node,
	                          Arguments inputs, Arguments indices,
	                          Arguments shape, Arguments memory,
	                          map<Node*, int> offsets,
	                          map<Node*, int> variables) = 0;

	void GenerateKernelLines(const IR* ir, const Lable* cluster,
	                         const Kernel* kernel) {
		NodeNames names = GenerateNodeNames(*ir);

		int indent = 1;
		int variable_index = 0;
		int memory_index = 0;
		// Translate each operation into HLSL
		for (auto node = IR::Iterator(cluster->node_);
		     !node.is_cluster_end(cluster); ++node) {

			if (node->name == "const") continue;

			if (node->name == "loop_end" || node->name == "if_end") {
				indent--;
			}

			// get node operation
			const Operation* op = node->op;

			// get node arguments
			Arguments inputs = node->GetArguments(Argument::Type::Input);
			Arguments indices = node->GetArguments(Argument::Type::Index);
			Arguments shape = node->GetArguments(Argument::Type::Shape);
			Arguments memory = node->GetArguments(Argument::Type::Memory);

			string name = names[*node];

			Line* line = GenerateLine(&names, op, node.get(), inputs, indices, shape,
			                         memory, kernel->memory, kernel->variables);
			line->indent = indent;
			lines.push_back(line);

			if (node->name == "loop_begin" || node->name == "if_begin") {
				indent++;
			}
		}
	}

	void Compactify()
	{
		//merge lines if short enough and rename variables
		map<string, Line*> line_map;
		map<string, vector<string>> line_inputs;
		map<string, vector<string>> line_outputs;
	
		for (auto& line : lines) {
			// get inputs
			line_inputs[line->name] = vector<string>();
			line_outputs[line->name] = vector<string>();
			
			for (auto& arg : line->arguments) {
				line_inputs[line->name].push_back(arg);
			}
	
			// add line to map
			line_map[line->name] = line;
		}

		for (auto& line : lines) {
			// get outputs
			for (auto& arg : line->arguments) {
				line_outputs[arg].push_back(line->name);
			}
		}
			
		unordered_set<Line*> toRemove;

		// merge lines
		const int max_line_length = 110;
		for (auto& line : lines) {
			int line_size = (int)line->expression.size();
			for (int i = 0; i < line->arguments.size(); i++) {
				string arg = line->arguments[i];
				Line* line2 = line_map[arg];
				int input_size = (int)line_map[arg]->expression.size();
				int output_count = (int)line_outputs[arg].size();
				if ((input_size + line_size < max_line_length && output_count == 1) ||
				    line2->cost < 1.0f) {
					// count the number of instances of arg in line->right
					std::regex arg_regex("\\b" + arg +
					                     "\\b");  // regex for whole word match
					auto words_begin = std::sregex_iterator(line->expression.begin(), line->expression.end(), arg_regex);
					auto words_end = std::sregex_iterator();
	
					int instances = (int)std::distance(words_begin, words_end);
	
					if (instances < 2 || line2->cost < 1.0f) {
						// merge lines
						string replace = line2->expression;

						if (line2->needs_parenthesis) {
							replace = "(" + replace + ")";
						}

						line->expression =
						    std::regex_replace(line->expression, arg_regex, replace);
	
						// add inputs
						for (auto& input : line_inputs[arg]) {
							if (input != arg) {
								line->arguments.push_back(input);
								line->cost += line_map[input]->cost;
							}
						}
	
						toRemove.insert(line2);  // Add the line to the removal set
					}
				}
			}
		}

		// remove lines
		for (auto& line : toRemove) {
			lines.remove(line);
			delete line;
		}
	}


	string GetFinalCode() {
		string code;
		int indent = 0;
		for (auto& line : lines) {
			for (int i = 0; i < line->indent; i++) {
				code += "  ";
			}
			code += line->left;
			code += line->expression;
			code += line->right;
			code += "\n";
		}

		// update names
		int i = 0;
		for (auto& line : lines) {
			string old_name = line->name;
			string new_name = "v" + to_string(i);
			i++;
			std::regex name_regex("\\b" + old_name +
			                      "\\b");  // regex for whole word match
			code = std::regex_replace(code, name_regex, new_name);
			line->name = new_name;
		}

		return code;
	}
};

class C_CodeGenerator : public CodeGenerator
{
public:
	Line* GenerateLine(NodeNames* names, const Operation* op, Node* node,
		Arguments inputs, Arguments indices,
		Arguments shape, Arguments memory, map<Node*, int> offsets, map<Node*, int> variables) override
	{
		// get node names
		vector<string> arguments;
		vector<string> input_variables;
		vector<DataType> input_types;
		for (const Argument& arg : memory) {
			string var_name = GetNodeName(arg.from_->get(), *names, true);
			arguments.push_back(var_name);
			if (arg.from_->get()->name != "const" && arg.from_->get()->name != "memory")
			{
				input_variables.push_back(var_name);
			}
		}
		for (const Argument& arg : indices) {
			string var_name = GetNodeName(arg.from_->get(), *names, true);
			arguments.push_back(var_name);
			if (arg.from_->get()->name != "const" &&
			    arg.from_->get()->name != "memory")
			{
				input_variables.push_back(var_name);
			}
		}
		for (const Argument& arg : inputs) {
			Node* input = arg.from_->get();
			string name = GetNodeName(input, *names, true);
			if (input->name == "memory") {
				name = "variables[" + to_string(variables[input]) + "]";
			}
			arguments.push_back(name);
			input_types.push_back(arg.from_->get()->tensor_->type);
			if (input->name != "const" && input->name != "memory")
			{
				input_variables.push_back(name);
			}
		}

		string name = (*names)[node];

		// get output type
		DataType output_type = op->GetOutputType(input_types);

		map<DataType, string> type_names = {
		    {DataType::None, "void"}, {DataType::Bool, "bool"}, {DataType::Float, "float"},
		    {DataType::Uint, "uint"}, {DataType::Int, "int"},
		};

		// generate line
		string left = "";
		string expression = "";
		string right = "";
		bool needs_parenthesis = true;
		if (op->name_ == "loop_begin") {
			left += "for (int " + name + " = " + arguments[0] + "; " + name + " < " +
			        arguments[1] + "; " + name + " += " + arguments[2] +
			        ") {";
		} else if (op->name_ == "loop_end") {
			left += "}";
		} else if (op->name_ == "if_begin") {
			left += "if (" + arguments[0] + ") {";
		} else if (op->name_ == "if_end") {
			left += "}";
		} else if (op->op_type_ == OpType::Store || op->op_type_ == OpType::Load) {
			string memory_expression = "memory[offsets[" +
			                           to_string(offsets[memory[0].from_->get()]) + "] + " +
			                           arguments[1] + "]";
			if (op->name_ == "load") {
				left += type_names[output_type] + " " + name + " = ";
				if (output_type == DataType::Float) {
					expression += "asfloat(";
				}
				if (output_type == DataType::Int) {
					expression += "asint(";
				}
				expression += memory_expression;
				if (output_type != DataType::Uint) {
					expression += ")";
				}
				right += ";";
				needs_parenthesis = false;
			} else if (op->name_ == "store") {
				expression += memory_expression + " = ";
				if (input_types[0] != DataType::Uint) {
					expression += "asuint(";
				}
				expression += arguments[2];
				if (input_types[0] != DataType::Uint) {
					expression += ")";
				}
				right += ";";
			}
		} else {
			if (output_type != DataType::None) {
				left += type_names[output_type] + " " + name + " = ";
			}
			string line;

			switch (op->op_type_) {
			case OpType::Operator:
				line += arguments[0] + " " + op->code_ + " " + arguments[1];
				break;
			case OpType::UnaryOperator:
				line += op->code_ + arguments[0];
				break;
			case OpType::Function:
				line += op->code_ + "(";
				for (int i = 0; i < arguments.size(); i++) {
					if (i != 0) {
						line += ", ";
					}
					line += arguments[i];
				}
				line += ")";
				needs_parenthesis = false;
				break;
			case OpType::Keyword:
				line += op->code_;
				break;
			case OpType::Variable:
				line += op->code_;
				needs_parenthesis = false;
				break;
			case OpType::TypeCast:
				line += "(" + op->code_ + ")" + arguments[0];
				break;
			case OpType::TypeReinterpret:
				line += "*(" + op->code_ + "*)&" + arguments[0];
				break;
			case OpType::Constant:
				line += node->tensor_->GetConstantString();
				needs_parenthesis = false;
				break;
			default:
				line += "";
				break;
			}
			expression += line;
			right += ";";
		}

		return new Line(left, expression, right, name, input_variables, needs_parenthesis, op->cost_);
	}
};

}  // namespace TensorFrost