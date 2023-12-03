#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

NodeNames GenerateNodeNames(const IR& ir) {
	NodeNames names = NodeNames();
	map<Lable*, int> cluster_var_index = map<Lable*, int>();
	int mem_index = 0;
	int cluster_index = 0;
	Lable* curent_cluster = nullptr;
	for (auto node = ir.begin(); !node.is_end(); ++node) {
		if (node->cluster_head_ != curent_cluster) {
			cluster_index++;
		}
		if (node->name == "memory") {
			names[*node] = "mem" + to_string(mem_index);
			mem_index++;
		} else {
			Lable* cluster_id = node->cluster_head_;
			int var_index = cluster_var_index[cluster_id];
			names[*node] =
			    "var" + to_string(cluster_index) + "_" + to_string(var_index);
			cluster_var_index[cluster_id]++;
		}
		curent_cluster = node->cluster_head_;
	}

	return names;
}

string GetNodeName(const Node* node, NodeNames& names, bool compact) {
	if (compact) {
		if (node->name == "const") {
			return node->GetTensor()->GetConstantString();
		}
	}
	return names[node];
}

inline string Tensor::GetConstantString() const {
	if (node_->name == "const" || node_->name == "dim_id") {
		switch (type) {
			case DataType::Float:
				return to_string(AsFloat(data[0])) + "f";
			case DataType::Int:
				return to_string(AsInt(data[0]));
			case DataType::Uint:
				return to_string(data[0]) + "u";
			default:
				return "";
		}
	} else {
		return "";
	}
}

void CodeGenerator::GenerateKernelLines(const IR* ir, const Lable* cluster,
                         const Kernel* kernel) {
	NodeNames names = GenerateNodeNames(*ir);

	int indent = 0;
	int variable_index = 0;
	int memory_index = 0;
	// Translate each operation into HLSL
	for (auto node = IR::Iterator(cluster->node_); !node.is_cluster_end(cluster);
	     ++node) {
		if (node->name == "const") continue;

		if (node->name == "loop_end" || node->name == "if_end") {
			indent--;
		}

		// get node operation
		const Operation* op = node->op;

		// get node arguments
		Arguments inputs = node->GetArguments(Arg::Type::Input);
		Arguments indices = node->GetArguments(Arg::Type::Index);
		Arguments shape = node->GetArguments(Arg::Type::Shape);
		Arguments memory = node->GetArguments(Arg::Type::Memory);

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

void CodeGenerator::Compactify() {
	// merge lines if short enough and rename variables
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
	const int max_line_length = 100;
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
				auto words_begin = std::sregex_iterator(
				    line->expression.begin(), line->expression.end(), arg_regex);
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

string CodeGenerator::GetFinalCode() {
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

string AddIndent(const string& input, const string& indent) {
	stringstream ss(input);
	string line;
	string indentedText;

	while (getline(ss, line)) {
		indentedText += indent + line + "\n";
	}

	return indentedText;
}

}  // namespace TensorFrost