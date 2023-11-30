#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

string GenerateKernelHLSL(const IR& ir, const Lable* cluster) {
	NodeNames names = GenerateNodeNames(ir);

	string hlsl_code;

	// Begin HLSL function
	hlsl_code += "void main() {\n";
	int indent = 1;
	// Translate each operation into HLSL
	for (auto node = IR::Iterator(cluster->node_); !node.is_cluster_end(cluster);
	     ++node) {
		if (node->name == "const") continue;

		if (node->name == "loop_end") {
			indent--;
		}

		// indent
		for (int i = 0; i < indent; i++) {
			hlsl_code += "  ";
		}

		// get node operation
		const Operation& op = FindOperation(node->name);

		// get node arguments
		Arguments inputs = node->GetArguments(Arg::Type::Input);
		Arguments indices = node->GetArguments(Arg::Type::Index);
		Arguments shape = node->GetArguments(Arg::Type::Shape);
		Arguments memory = node->GetArguments(Arg::Type::Memory);

		// check number of indices
		if (indices.size() > 1) {
			throw std::runtime_error(
			    "HLSL codegen does not support multidimensional indexing");
		}

		// get node names
		vector<string> arguments;
		vector<DataType> input_types;
		for (const Arg& arg : memory) {
			arguments.push_back(GetNodeName(arg.from_->get(), names, true));
		}
		for (const Arg& arg : inputs) {
			arguments.push_back(GetNodeName(arg.from_->get(), names, true));
			input_types.push_back(arg.from_->get()->tensor_->type);
		}
		for (const Arg& arg : indices) {
			arguments.push_back(GetNodeName(arg.from_->get(), names, true));
		}

		hlsl_code += op.GenerateLine(names[*node], arguments, input_types) + "\n";

		if (node->name == "loop_begin") {
			indent++;
		}
	}

	// End HLSL function
	hlsl_code += "}\n";

	return hlsl_code;
}

string GenerateHLSL(const IR& ir) {
	string all_kernels;

	ClusterProp clusters = ir.GetClusterProperties();

	// Generate HLSL code for each cluster
	int kernel_count = 0;
	for (auto* cluster : clusters.cluster_heads) {
		if (cluster->node_->name == "memory") continue;

		// Check if cluster has shape
		if (cluster->node_->GetArguments(Arg::Type::Shape).empty()) continue;

		// Generate kernel
		all_kernels += "Kernel ID: " + to_string(kernel_count++) + "\n";
		all_kernels += GenerateKernelHLSL(ir, cluster);
		all_kernels += "\n";
	}

	return all_kernels;
}
}  // namespace TensorFrost