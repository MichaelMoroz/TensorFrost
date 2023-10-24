#pragma once

#include "IR/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

string GenerateHLSL(const IR& ir) {
    list<Tensor*> nodes = ir.GetNodes();

	// first give unique names to all the tensors
	TensorNames names = TensorNames();
	int index = 0;
	for (const Tensor* node : nodes) 
	{
		names[node] = "var" + to_string(index);
		index++;
	}

    string hlslCode;
    
    // Begin HLSL function
    hlslCode += "void main() {\n";
	int indent = 1;
    // Translate each operation into HLSL
    for (const Tensor* node : nodes) {
		if (node->name == "const") continue;

		if (node->name == "loop_end") {
			indent--;
		}

		// indent
		for (int i = 0; i < indent; i++) {
			hlslCode += "  ";
		}

		//get node operation
		const Operation& op = FindOperation(node->name);

		//get node arguments
		Arguments inputs = node->GetArguments(Argument::Type::Input);
		Arguments indices = node->GetArguments(Argument::Type::Index);
		Arguments shape = node->GetArguments(Argument::Type::Shape);

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

		hlslCode += op.GenerateLine(names[node], arguments, input_types) + "\n";

		if (node->name == "loop_begin") {
			indent++;
		}
    }
    
    // End HLSL function
    hlslCode += "}\n";
    
    return hlslCode;
}

}