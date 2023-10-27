#pragma once

#include "IR/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

string GenerateHLSL(const IR& ir) {
	// first give unique names to all the tensors
	NodeNames names = NodeNames();
	int index = 0;
	for (auto node = ir.begin(); !node.is_end(); ++node)
	{
		names[*node] = "var" + to_string(index);
		index++;
	}

    string hlslCode;
    
    // Begin HLSL function
    hlslCode += "void main() {\n";
	int indent = 1;
    // Translate each operation into HLSL
	for (auto node = ir.begin(); !node.is_end(); ++node) {
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

        //check number of indices
        if(indices.size() > 1)
        {
            throw std::runtime_error("HLSL codegen does not support multidimensional indexing");
        }

		//get node names
		vector<string> arguments;
		vector<DataType> input_types;
		for (const Argument& arg : inputs) {
			      arguments.push_back(GetNodeName(arg.from_->get(), names, true));
			      input_types.push_back(arg.from_->get()->tensor_->type);
		}
		for (const Argument& arg : indices) {
			      arguments.push_back(GetNodeName(arg.from_->get(), names, true));
		}

		hlslCode +=
		    op.GenerateLine(names[*node], arguments, input_types) + "\n";

		if (node->name == "loop_begin") {
			indent++;
		}
    }
    
    // End HLSL function
    hlslCode += "}\n";
    
    return hlslCode;
}

}