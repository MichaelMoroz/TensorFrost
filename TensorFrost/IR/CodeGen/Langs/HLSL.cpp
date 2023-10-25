#pragma once

#include "IR/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

string GenerateHLSL(const IR& ir) {
	list<const Node*> nodes = ir.GetNodes();

	// first give unique names to all the tensors
	NodeNames names = NodeNames();
	int index = 0;
	for (const Node* node : nodes) 
	{
		names[node] = "var" + to_string(index);
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
		Arguments inputs = node->GetArguments(Argument::Type::Input);
		Arguments indices = node->GetArguments(Argument::Type::Index);
		Arguments shape = node->GetArguments(Argument::Type::Shape);

		//get node names
		vector<string> arguments;
		vector<DataType> input_types;
		for (const Argument& arg : inputs) {
			arguments.push_back(GetNodeName(arg.node_, names, true));
			input_types.push_back(arg.node_->tensor_->type);
		}
		for (const Argument& arg : indices) {
			arguments.push_back(GetNodeName(arg.node_, names, true));
		}

		hlslCode +=
		    op.GenerateLine(names[node], arguments, input_types) + "\n";

		if (node->tensor_->name == "loop_begin") {
			indent++;
		}
    }
    
    // End HLSL function
    hlslCode += "}\n";
    
    return hlslCode;
}

}