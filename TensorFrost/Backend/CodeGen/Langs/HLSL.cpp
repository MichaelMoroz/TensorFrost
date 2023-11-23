#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

string GenerateKernelHLSL(const IR& ir, const Lable* cluster) {
	NodeNames names = GenerateNodeNames(ir);

    string hlslCode;
    
    // Begin HLSL function
    hlslCode += "void main() {\n";
	int indent = 1;
    // Translate each operation into HLSL
	for (auto node = IR::iterator(cluster->node_); !node.is_cluster_end(cluster);
	     ++node) {
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
		Arguments memory = node->GetArguments(Argument::Type::Memory);

        //check number of indices
        if(indices.size() > 1)
        {
			throw std::runtime_error("HLSL codegen does not support multidimensional indexing");
        }

		//get node names
		vector<string> arguments;
		vector<DataType> input_types;
		for (const Argument& arg : memory) {
			      arguments.push_back(GetNodeName(arg.from_->get(), names, true));
		}
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

string GenerateHLSL(const IR& ir) {
	string allKernels = "";

	ClusterProp clusters = ir.GetClusterProperties();
	
	// Generate HLSL code for each cluster
	int kernel_count = 0;
	for (auto cluster : clusters.cluster_heads) {
		if (cluster->node_->name == "memory") continue;

		// Check if cluster has shape 
		if (cluster->node_->GetArguments(Argument::Type::Shape).size() == 0) continue;

		// Generate kernel
		allKernels += "Kernel ID: " + to_string(kernel_count++) + "\n";
		allKernels += GenerateKernelHLSL(ir, cluster);
		allKernels += "\n";
	}
	
	return allKernels;

}
}  // namespace TensorFrost