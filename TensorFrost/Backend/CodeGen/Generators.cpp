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

unordered_set<string> forbidden_names = {
	"unsigned", "input", "output", "max", "min", "exp", "sin", "cos", "if", "else", "while", "for", "switch", "case", "default", "break",
    "this",  "true", "false", "null", "new", "delete", "return", "continue", "goto", "try", "catch", "throw", 
	"const", "static", "extern", "inline", "virtual", "override", "final", "public", "protected", "private", "sample",
	"texture", "sampler", "uniform", "varying", "attribute", "in", "out", "inout", "layout", "precision", "highp",
	"mediump", "lowp", "noperspective", "flat", "smooth", "centroid", "patch", "sample", "subroutine", "common",
	"partition", "active", "asm", "class", "union", "enum", "typedef", "template", "typename", "using", "namespace",
	"friend", "sizeof", "alignof", "typeid", "dynamic_cast", "static_cast", "const_cast", "reinterpret_cast", "sizeof",
	"alignof", "typeid", "noexcept", "throw", "auto", "register", "explicit", "mutable", "thread_local",
	"constexpr", "decltype", "noexcept", "nullptr", "alignas", "and", "and_eq", "bitand", "bitor", "compl", "not", "not_eq", "or",
	"sign", "xor", "xor_eq", "bool", "break", "case", "char"
};

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
		if (strip_debug_names) {
			node->debug_name = "";
		}
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

string GetGroupBufferDeclarations(Kernel *kernel, function<string(const string &, const string &, int)> get_shared_name) {
	string final_source;

	// add group memory declarations
	for (auto& mem : kernel->group_memory) {
		string name = mem->var_name;
		string type_name = type_names[mem->format.type];
		//TODO: add support for non 32 bit types
		final_source += get_shared_name(name, type_name, mem->data[0]);
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
	string name = node->var_name;
	if (compact) {
		if (node->name == "const" && !node->flags.has(NodeProp::Modified)) {
			name = node->GetTensor()->GetConstantString();
		}
	}
	else {
		if (node->name == "const") {
			name = node->var_name + "(" + node->GetTensor()->GetConstantString() + ")";
		}
	}
	if (name.empty()) {
		name = node->name + "_" + to_string(node->debug_index);
	}
	return name;
}

#ifdef HAS_FORMAT
string format_float(double x) {
	std::string s = std::format("{}", x);
	if (s.find('.') == std::string::npos && s.find('e') == std::string::npos) {
		s += '.';
	}
	return s + 'f';
}
#else
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
#endif

inline string Tensor::GetConstantString() const {
	if (node_->name == "const" || node_->name == "dim_id") {
		switch (node_->format.type) {
			case TFType::Float:
				return format_float(AsFloat(node_->data[0]));
			case TFType::Int:
				return to_string(AsInt(node_->data[0]));
			case TFType::Uint:
				return to_string(node_->data[0]) + "u";
			case TFType::Bool:
				return node_->data[0] == 0 ? "false" : "true";
			default:
				throw std::runtime_error("Unsupported constant type");
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

Line* CodeGenerator::GenerateLine(Node* node)  {
	ArgumentManager& args = node->args;
	if (kernel) RegenerateNodeName(node);
	GenerateArgumentNames(args);
	const Operation* op = node->op;

	string name = node->var_name;

	// get output type
	TFDataFormat output_format = node->format;
	//TODO: add support for non 32 bit types

	// generate line
	string left = "";
	string expression = "";
	string right = "";
	bool needs_paranthesis = false;

	if (op->HasAllTypes(OpProp::Special)) {
		int dims = args.Count(ArgType::Shape);

		string shape_arg = "{";

		for (int j = 0; j < dims; j++) {
			if (j != 0) {
				shape_arg += ", ";
			}
			Node* shape_node = args.Get(ArgType::Shape, j);

			shape_arg += "(uint)" + args.Name(ArgType::Shape, dims - j - 1);
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
				left += "tf.check_tensor(" + node->var_name+ ", \"" + node->var_name + "\", " + shape_arg + ", " + DataFormatNames[output_format] + ")";
				right += ";";
			}
			// if any other memory type - allocate it
			else {
				left += "TFTensor " + node->var_name + " = ";
				expression += "tf.allocate(\"" + node->var_name + "\", " + shape_arg + ", " + DataFormatNames[output_format] + ")";
				right += ";";
			}
		} else if (op->name_ == "deallocate") {
			left = "tf.deallocate(" + args.Name(ArgType::Memory) + ")";
			right = ";";
		} else if (op->name_ == "input_shape") {
			left = "int " + node->var_name + " = ";
			expression = ir->input_memory_map[(int)node->flags.get(NodeProp::InputShapeMemory)]->var_name + ".shape[" + to_string((int)node->flags.get(NodeProp::InputShapeDim)) + "]";
			right = ";";
		} else if(op->HasAllTypes(OpProp::MemoryReuse)) {
			left = "TFTensor " + node->var_name + " = ";
			expression = "tf." + op->code_ + "(" + args.Name(ArgType::Memory) + ", \"" + node->var_name + "\", " + shape_arg + ", " + DataFormatNames[output_format] + ")";
			right = ";";
		} else if(op->HasAllTypes(OpProp::Debug)) {
			left = "tf." + op->code_ + "(\"" + node->debug_name + "\"";
			if (args.Has(ArgType::Input)) {
				left += ", " + args.Name(ArgType::Input);
			}
			left += ")";
			right = ";";
		} else if(op->name_ == "local_memory") {
			left = type_names[output_format.type] + " " + name + "[" + to_string(node->data[0]) + "]";
			right = ";";
		} else if(op->name_ == "group_memory") {
			left = "";
			//just leave as comment, actual declaration is done outside of the main body of the kernel
			right = "//" + type_names[output_format.type] + " " + name + "[" + to_string(node->data[0]) + "]";
		}
	} else if (op->HasAllTypes(OpProp::MemoryOp)) {
		string address;

		if (kernel) {
			address = "0";
			// if has index (not a scalar)
			if (args.Has(ArgType::Index)) {
				address = args.Name(ArgType::Index);
			}

			bool is_local = node->flags.has(NodeProp::LocalMemoryOp);
			string memory_name = args.Name(ArgType::Memory) + (is_local ? "" : "_mem");
			string memory_expression = memory_name + "[" + address + "]";
			TFType memory_type = is_local ? node->format.type : Uint;
			string memory_type_name = type_names[memory_type];

			if (op->name_ == "load") {
				string output_type_name = type_names[output_format.type];
				left += output_type_name + " " + name + " = ";
				expression +=
				    (output_format.type == memory_type)
				        ? memory_expression
				        : TypeReinterpret(output_type_name, memory_expression);
				right += ";";
			} else if (op->name_ == "store") {
				expression += memory_expression + " = ";
				expression +=
				    (output_format.type == memory_type)
				        ? args.Name(ArgType::Input)
				        : TypeReinterpret(memory_type_name, args.Name(ArgType::Input));
				right += ";";
			} else if (op->HasAllTypes(OpProp::Scatter)) {
				if (output_format.type != None) {
					left += type_names[output_format.type] + " " + name + " = ";
				}
				string output_type_name = type_names[output_format.type];
				string input_type_name = type_names[args.Format(ArgType::Input).type];
				expression += GenerateAtomicOp(op->name_, input_type_name,
				                               output_type_name, address,
				                               args.Name(ArgType::Input), name, memory_name);
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
				string output_type_name = type_names[output_format.type];
				left += output_type_name + " " + name + " = ";
				string memory_expression = GetName("tf.read") + "(" + tensor_name + ", " + address + ")";
				expression += (output_format.type == Uint)
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
		if (output_format.type != None) {
			left += type_names[output_format.type] + " " + name + " = ";
		}
		string line;
		string code = op->code_;
		switch (op->class_) {
			case OpClass::Operator:
				args.AddParenthesis(true);
				if ((code == "&" || code == "|") && output_format.type == Bool) {
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
				line += GenerateTypeCast(&args, DataTypeToString(node->format.type));
				break;
			case OpClass::TypeReinterpret:
				line += GenerateTypeReinterpret(&args, DataTypeToString(node->format.type));
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