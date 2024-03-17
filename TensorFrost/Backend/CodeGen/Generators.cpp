#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

void GenerateCode(Program* program) {
	program->generated_code_ = GenerateCPP(program);
}

void GenerateNodeNames(const IR& ir) {
	int var_index = 0;
	int mem_index = 0;
	int cluster_index = 0;
	Node* curent_cluster = nullptr;
	for (auto node = ir.begin(); !node.end(); node.next()) {
		if (node->parent != curent_cluster) {
			cluster_index++;
			var_index = 0;
		}
		if (node->name == "memory") {
			node->var_name = "m" + to_string(mem_index);
			mem_index++;
		} else {
			node->var_name =
			    "v" + to_string(cluster_index) + "_" + to_string(var_index);
			var_index++;
		}
		curent_cluster = node->parent;
	}
}

string ReadVariable(Node* node) {
	if (node->name == "const") {
		return to_string(node->GetTensor()->data[0]);
	}
	if (node->name == "memory") {
		return "mem[" + node->var_name + "]";
	}
	return node->var_name;
}

string GetNodeName(const Node* node,  bool compact) {
	if (compact) {
		if (node->name == "const" && !node->has_been_modified_) {
			return node->GetTensor()->GetConstantString();
		}
	}
	else {
		if (node->name == "const") {
			return node->var_name + "(" + node->GetTensor()->GetConstantString() + ")";
		}
	}
	return node->var_name;
}

std::string format_float(float x) {
	std::string s = std::format("{}", x);
	if (s.find('.') == std::string::npos && s.find('e') == std::string::npos) {
		s += '.';
	}
	return s + 'f';
}

inline string Tensor::GetConstantString() const {
	if (node_->name == "const" || node_->name == "dim_id") {
		switch (type) {
			case DataType::Float:
				return format_float(AsFloat(data[0]));
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

void CodeGenerator::GenerateKernelLines(const IR* ir, const Kernel* kernel) {
	int variable_index = 0;
	int memory_index = 0;
	int prev_depth = 0;
	// Translate each operation into HLSL
	for (auto node = NodeIterator(kernel->node_); !node.end(); node->name == "kernel" ? node.forward() : node.next()) {
		if (node->name == "const" && !node->has_been_modified_) {
			continue;
		}

		string name = node->var_name;

		int depth = node.depth() - 1;
		if (depth != prev_depth) {
			// add scope brackets
			if (depth < prev_depth) {
				for (int i = prev_depth - 1; i >= depth; i--) {
					lines.push_back(new Line(i, "}"));
				}
			} else if (depth > prev_depth) {
				for (int i = prev_depth; i < depth; i++) {
					lines.push_back(new Line(i, "{"));
				}
			}
		}

		Line* line = nullptr;
		if (custom_generated_code_.contains(*node)) {
			line = new Line("", custom_generated_code_[*node], ";", "", false, 0);
		} else {
			// get node arguments
			line = GenerateLine(*node, kernel->memory, kernel->variables);
		}

		if (line == nullptr) {
			continue;
		}
		
		line->indent = depth;
		lines.push_back(line);
		prev_depth = depth;
	}

	// add closing brackets
	for (int i = prev_depth - 1; i >= 0; i--) {
		lines.push_back(new Line(i, "}"));
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
	//int i = 0;
	//for (auto& line : lines) {
	//	string old_name = line->name;
	//	string new_name = "v" + to_string(i);
	//	i++;
	//	std::regex name_regex("\\b" + old_name +
	//	                      "\\b");  // regex for whole word match
	//	code = std::regex_replace(code, name_regex, new_name);
	//	line->name = new_name;
	//}

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