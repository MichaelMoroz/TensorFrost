#ifndef TENSORFROST_BACKEND_CODEGEN_GENERATORS_CPP
#define TENSORFROST_BACKEND_CODEGEN_GENERATORS_CPP

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

void GenerateKernel(Program* program, Kernel* kernel) {
	switch (current_kernel_lang) {
		case CodeGenLang::CPP:
			GenerateCPPKernel(program, kernel);
			return;
		case CodeGenLang::HLSL:
			GenerateHLSLKernel(program, kernel);
			return;
		case CodeGenLang::GLSL:
			GenerateGLSLKernel(program, kernel);
			return;
		default:
			throw std::runtime_error("Code generation for this language is not implemented yet");
	}
}

unordered_set<string> forbidden_names = {"unsigned", "input", "output", "max", "min", "exp", "sin", "cos", "if", "else", "while", "for", "switch", "case", "default", "break",
    "this",  "true", "false", "null", "new", "delete", "return", "continue", "goto", "try", "catch", "throw", 
	"const", "static", "extern", "inline", "virtual", "override", "final", "public", "protected", "private"};

bool IsForbiddenName(const string& name) {
	return forbidden_names.contains(name);
}

void GenerateNodeNames(const IR& ir) {
	int var_index = 0;
	int mem_index = 0;
	int cluster_index = 0;
	Node* curent_cluster = nullptr;
	map<string, int> name_count;
	for (auto node = ir.begin(); !node.end(); node.next()) {
		if (node->parent != curent_cluster) {
			cluster_index++;
			var_index = 0;
		}
		string debug = node->debug_name;
		if (!debug.empty()) {
			// check if the name is already used
			if (name_count.contains(debug)) {
				name_count[debug]++;
				debug = debug + "_" + to_string(name_count[debug]);
			}
			else {
				name_count[debug] = 1;
			}
			if (IsForbiddenName(debug) ) {
				debug = debug + "0";
			}
			node->var_name = debug;
		} 
		else
		{
			if (node->name == "memory") {
				node->var_name = debug + "m" + to_string(mem_index);
				mem_index++;
			} else {
				node->var_name =
				    debug + "v" + to_string(cluster_index) + "_" + to_string(var_index);
				var_index++;
			}
		}

		curent_cluster = node->parent;
	}
}


string GetBufferDeclarations(Kernel *kernel, function<string(const string &, const string &, size_t)> get_name) {
	map<Node*, size_t> memory_bindings = kernel->GetMemoryBindings();

	vector<string> buffer_declarations = vector<string>(memory_bindings.size());
	for (auto& buffer : memory_bindings) {
		Node* mem_node = buffer.first;
		size_t binding = buffer.second;
		string name = mem_node->var_name;
		string type_name = "uint";
		buffer_declarations[binding] = get_name(name, type_name, binding);
	}

	string final_source;
	for (auto& decl : buffer_declarations) {
		final_source += decl;
	}
	return final_source;
}

string ReadVariable(Node* node) {
	if (node->name == "const") {
		return to_string(node->data[0]);
	}
	if (node->name == "memory") {
		return "mem[" + node->var_name + "]";
	}
	return node->var_name;
}

string GetNodeName(const Node* node,  bool compact) {
	if (compact) {
		if (node->name == "const" && !node->HasFlags(NodeFlags::Modified)) {
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

//std::string format_float(float x) {
//	std::string s = std::format("{}", x);
//	if (s.find('.') == std::string::npos && s.find('e') == std::string::npos) {
//		s += '.';
//	}
//	return s + 'f';
//}

string format_float(double value) {
	std::ostringstream out;

	// Determine when to use scientific notation vs fixed
	bool use_scientific = std::abs(value) < 1e-4 || std::abs(value) > 1e6;
	//and if not a zero value
	use_scientific = use_scientific && value != 0.0;
	if (use_scientific) {
		out << std::scientific;  // Use scientific notation for very small or large
		                         // numbers
	} else {
		out << std::fixed;  // Use fixed notation for moderate values
	}

	out << std::setprecision(7) << value;

	// Convert to string
	std::string str = out.str();

	// Remove trailing zeros and potentially unnecessary decimal point
	size_t endpos = str.find_last_not_of('0');
	if (endpos != std::string::npos) {
		str = str.substr(0, endpos + 1);
	}
	if (str.back() == '.') {
		str.pop_back();
	}

	// remove all zeros before "e"
	size_t epos = str.find('e');
	if (epos != std::string::npos) {
		size_t startpos = str.find_last_not_of('0', epos - 1);
		if (startpos != std::string::npos) {
			str = str.substr(0, startpos + 1) + str.substr(epos);
		}
	}

	if (str.find('.') == string::npos && str.find('e') == string::npos) {
		str += '.';
	}

	// add a zero digit after the decimal point if the next character is not a
	// digit
	size_t dotpos = str.find('.');
	if (dotpos != std::string::npos && !isdigit(str[dotpos + 1])) {
		//add a zero digit after the decimal point
		str.insert(dotpos + 1, "0");
	}
	
	return str + 'f';
}

inline string Tensor::GetConstantString() const {
	if (node_->name == "const" || node_->name == "dim_id") {
		switch (node_->type) {
			case TFType::Float:
				return format_float(AsFloat(node_->data[0]));
			case TFType::Int:
				return to_string(AsInt(node_->data[0]));
			case TFType::Uint:
				return to_string(node_->data[0]) + "u";
			default:
				return "";
		}
	} else {
		return "";
	}
}

void CodeGenerator::GenerateKernelCode(Kernel* kernel_) {
	kernel = kernel_;
	variables = kernel->variables;
	read_write_bindings = kernel->read_write_memory;
	read_only_bindings = kernel->read_only_memory;
	GenerateCode(kernel->root);
}

void CodeGenerator::GenerateCode(const Node* root) {
	int variable_index = 0;
	int memory_index = 0;
	int prev_depth = 0;
	// Translate each operation into HLSL
	for (auto node = NodeIterator(root); !node.end(); node->name == "kernel" ? node.forward() : node.next()) {
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
			line = new Line(*node, "", custom_generated_code_[*node], ";", "");
		} else {
			// get node arguments
			line = GenerateLine(*node);
		}

		if (line == nullptr) {
			continue;
		}
		
		line->indent = depth;
		lines.push_back(line);

		for (auto additional: additional_lines) {
			lines.push_back(new Line(depth, additional));
		}
		additional_lines.clear();

		prev_depth = depth;
	}

	// add closing brackets
	for (int i = prev_depth - 1; i >= 0; i--) {
		lines.push_back(new Line(i, "}"));
	}

	//remove lines
	unordered_set<Line*> remove_lines;
	for (auto& line : lines) {
		if (lines_to_remove.contains(line->node)) {
			remove_lines.insert(line);
		}
	}

	for (auto& line : remove_lines) {
		lines.erase(std::remove(lines.begin(), lines.end(), line), lines.end());
	}
}


string CodeGenerator::AssembleString() {
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

#endif