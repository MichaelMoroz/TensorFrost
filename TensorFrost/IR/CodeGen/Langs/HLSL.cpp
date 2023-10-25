#pragma once

#include "IR/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

string GenerateHLSL(const IR& ir) {
	list<const Node*> nodes = ir.GetNodes();

	// first give unique names to all the tensors
	TensorNames names = TensorNames();
	int index = 0;
	for (const Node* node : nodes) 
	{
		names[node->tensor_] = "var" + to_string(index);
		index++;
	}

    string hlslCode;
    
    // Begin HLSL function
    hlslCode += "void main() {\n";
	int indent = 1;
    // Translate each operation into HLSL
	for (const Node* node : nodes) {
		if (node->tensor_->name == "const") continue;

		if (node->tensor_->name == "loop_end") {
			indent--;
		}

		// indent
		for (int i = 0; i < indent; i++) {
			hlslCode += "  ";
		}

		//get node operation
		const Operation& op = FindOperation(node->tensor_->name);

		//get node arguments
		Arguments inputs = node->tensor_->GetArguments(Argument::Type::Input);
		Arguments indices = node->tensor_->GetArguments(Argument::Type::Index);
		Arguments shape = node->tensor_->GetArguments(Argument::Type::Shape);

		//get node names
		vector<string> arguments;
		vector<DataType> input_types;
		for (const Argument& arg : inputs) {
			arguments.push_back(GetNodeName(arg.tensor, names, true));
			input_types.push_back(arg.tensor->type);
		}
		for (const Argument& arg : indices) {
			arguments.push_back(GetNodeName(arg.tensor, names, true));
		}

		hlslCode +=
		    op.GenerateLine(names[node->tensor_], arguments, input_types) + "\n";

		if (node->tensor_->name == "loop_begin") {
			indent++;
		}
    }
    
    // End HLSL function
    hlslCode += "}\n";
    
    return hlslCode;
}

}