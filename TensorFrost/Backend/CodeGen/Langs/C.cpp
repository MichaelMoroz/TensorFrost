#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

string GenerateKernelC(const IR* ir, const Lable* cluster,
                       const Kernel* kernel,
                       const string& kernel_name) {
	//NodeNames names = GenerateNodeNames(*ir);

	string hlsl_code;

	// Begin HLSL function
	hlsl_code += "void " + kernel_name + "(";
	hlsl_code += "uint* variables, ";
	hlsl_code += "uint* offsets, ";
	hlsl_code += "uint* memory, ";
	hlsl_code += "uint thread_id";
	hlsl_code += ")\n";
	hlsl_code += "{\n";

	// Indentation level
	//int indent = 1;
	//int variable_index = 0;
	//int memory_index = 0;
	//// Translate each operation into HLSL
	//for (auto node = IR::Iterator(cluster->node_); !node.is_cluster_end(cluster);
	//     ++node) {
	//	if (node->name == "const") continue;
	//
	//	if (node->name == "loop_end") {
	//		indent--;
	//	}
	//
	//	// indent
	//	for (int i = 0; i < indent; i++) {
	//		hlsl_code += "  ";
	//	}
	//
	//	// get node operation
	//	const Operation& op = FindOperation(node->name);
	//
	//	// get node arguments
	//	Arguments inputs = node->GetArguments(Argument::Type::Input);
	//	Arguments indices = node->GetArguments(Argument::Type::Index);
	//	Arguments shape = node->GetArguments(Argument::Type::Shape);
	//	Arguments memory = node->GetArguments(Argument::Type::Memory);
	//
	//	// check number of indices
	//	if (indices.size() > 1) {
	//		throw std::runtime_error(
	//		    "Codegen does not support multidimensional indexing");
	//	}
	//
	//	// get node names
	//	vector<string> arguments;
	//	vector<DataType> input_types;
	//	for (const Argument& arg : memory) {
	//		arguments.push_back(GetNodeName(arg.from_->get(), names, true));
	//	}
	//	for (const Argument& arg : indices) {
	//		arguments.push_back(GetNodeName(arg.from_->get(), names, true));
	//	}
	//	for (const Argument& arg : inputs) {
	//		Node* input = arg.from_->get();
	//		string name = GetNodeName(input, names, true);
	//		if (input->name == "memory") {
	//			name = "variables[" + to_string(variable_index++) + "]";
	//		}
	//		arguments.push_back(name);
	//		input_types.push_back(arg.from_->get()->tensor_->type);
	//	}
	//
	//	string name = names[*node];
	//
	//	if (node->op->GetOpType() == OpType::Store ||
	//	    node->op->GetOpType() == OpType::Load) {
	//		arguments[0] = "memory";
	//		arguments[1] =
	//		    "offsets[" + to_string(memory_index++) + "] + " + arguments[1];
	//	}
	//
	//	hlsl_code += op.GenerateLine(names[*node], arguments, input_types) + "\n";
	//
	//	if (node->name == "loop_begin") {
	//		indent++;
	//	}
	//}

	C_CodeGenerator generator;
	generator.GenerateKernelLines(ir, cluster, kernel);
	//generator.Compactify();

	hlsl_code += generator.GetFinalCode();

	// End HLSL function
	hlsl_code += "}\n";

	return hlsl_code;
}

pair<string, vector<string>> GenerateC(Program* program) {
	string all_kernels =
	    "#include <cmath> \n"
	    "extern \"C\" \n"
	    "{ \n"
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

	// Generate HLSL code for each cluster
	int kernel_count = 0;
	vector<string> kernel_names;
	for (auto& i : program->kernels_) {
		Kernel* kernel = &i;
		Lable* cluster = kernel->begin_->cluster_head_;
		if (kernel->type_ != KernelType::Compute) {
			continue;
		}

		string kernel_name = "kernel_" + to_string(kernel_count++);
		string function_name = kernel_name + "_execute";
		kernel_names.push_back(function_name);

		// Generate kernel
		all_kernels += "\n";
		string kernel_code = GenerateKernelC(program->ir_, cluster, kernel, kernel_name);
		kernel->generate_code_ = kernel_code;
		all_kernels += kernel_code;
		all_kernels += "\n";
		all_kernels += "__declspec(dllexport) void " + function_name + "(";
		all_kernels += "uint* variables, ";
		all_kernels += "uint* offsets, ";
		all_kernels += "uint* memory, ";
		all_kernels += "uint threads";
		all_kernels += ")\n";
		all_kernels += "{\n";
		all_kernels += "  #pragma omp parallel for\n";
		all_kernels += "  for(int i = 0; i < threads; i++)\n";
		all_kernels += "  {\n";
		all_kernels += "    " + kernel_name + "(variables, offsets, memory, i);\n";
		all_kernels += "  }\n";
		all_kernels += "}\n";
		all_kernels += "\n";
	}

	all_kernels += "}\n";

	program->generate_code_ = all_kernels;
	return pair<string, vector<string>>(all_kernels, kernel_names);
}
}  // namespace TensorFrost