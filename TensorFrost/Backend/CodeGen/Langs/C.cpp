#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

string GenerateKernelC(const IR& ir, const Lable* cluster, const string kernel_name) {
	NodeNames names = GenerateNodeNames(ir);

    string hlslCode;
    
    // Begin HLSL function
    hlslCode += "void " + kernel_name + "(";
	hlslCode += "uint* variables, ";
	hlslCode += "uint* offsets, ";
	hlslCode += "uint* memory, ";
	hlslCode += "uint thread_id";
	hlslCode += ")\n";
	hlslCode += "{\n";
	
	// Indentation level
	int indent = 1;
	int variable_index = 0;
	int memory_index = 0;
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
			throw std::runtime_error("Codegen does not support multidimensional indexing");
        }

		//get node names
		vector<string> arguments;
		vector<DataType> input_types;
		for (const Argument& arg : memory) {
			arguments.push_back(GetNodeName(arg.from_->get(), names, true));
		}
		for (const Argument& arg : indices) {
			arguments.push_back(GetNodeName(arg.from_->get(), names, true));
		}
		for (const Argument& arg : inputs) {
			Node* input = arg.from_->get();
			string name = GetNodeName(input, names, true);
			if(input->name == "memory")
			{
				name = "variables[" + to_string(variable_index++) + "]";
			}
			arguments.push_back(name);
			input_types.push_back(arg.from_->get()->tensor_->type);
		}

        string name = names[*node];

        if(node->op->GetOpType() == OpType::Store || node->op->GetOpType() == OpType::Load)
        {
            arguments[0] = "memory";
            arguments[1] = "offsets[" + to_string(memory_index++) + "] + " + arguments[1];
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

pair<string, vector<string>> GenerateC(const IR& ir) {
	string allKernels =
	      "#include <math.h> \n"
	      //"extern \"C\" \n"
		//"{ \n"
	      "typedef unsigned int uint; \n"
	      "\n"
	      "float asfloat(uint x) \n"
	      "{ \n"
	      "  return *(float*)&x; \n"
	      "} \n"
	      "\n"
	      "uint asuint(float x) \n"
	      "{ \n"
	      "  return *(uint*)&x; \n"
	      "} \n"
		  "\n"
		  "int clamp(int x, int min, int max) \n"
		  "{ \n"
		  "  if(x < min) return min; \n"
		  "  if(x > max) return max; \n"
		  "  return x; \n"
		  "} \n";

	ClusterProp clusters = ir.GetClusterProperties();
	
	// Generate HLSL code for each cluster
	int kernel_count = 0;
	vector<string> kernel_names;
	for (auto cluster : clusters.cluster_heads) {
		if (cluster->node_->name == "memory") continue;

		// Check if cluster has shape 
		if (cluster->node_->GetArguments(Argument::Type::Shape).size() == 0) continue;

		string kernel_name = "kernel_" + to_string(kernel_count++);
		string function_name = kernel_name + "_execute";
		kernel_names.push_back(function_name);

		// Generate kernel
		allKernels += "\n";
		allKernels += GenerateKernelC(ir, cluster, kernel_name);
		allKernels += "\n";
		allKernels += "__declspec(dllexport) void " + function_name + "(";
		allKernels += "uint* variables, ";
		allKernels += "uint* offsets, ";
		allKernels += "uint* memory, ";
		allKernels += "uint threads";
		allKernels += ")\n";
		allKernels += "{\n";
		allKernels += "  for(uint i = 0; i < threads; i++)\n";
		allKernels += "  {\n";
		allKernels += "    " + kernel_name + "(variables, offsets, memory, i);\n";
		allKernels += "  }\n";
		allKernels += "}\n";
		allKernels += "\n";
	}
	

	//allKernels += "}\n";
	return pair<string, vector<string>>(allKernels, kernel_names);

}
}  // namespace TensorFrost