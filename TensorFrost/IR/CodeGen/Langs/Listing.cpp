#pragma once

#include "IR/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

string GetNodeName(const Node* node, NodeNames& names, bool compact) {
	if (compact)
	{
		if (node->name == "const") {
			return node->tensor_->GetConstantString();
		}
	}
	return names[node];
}

inline string Tensor::GetConstantString() const {
	if (node->name == "const" || node->name == "dim_id") {
		switch (type) {
			case DataType::Float:
				return to_string(AsFloat(data[0]));
			case DataType::Int:
				return to_string(AsInt(data[0]));
			case DataType::Uint:
				return to_string(data[0]);
			default:
				return "";
		}
	} else {
		return "";
	}
}

string GetOperationListing(const IR& ir, bool compact) 
{
	// first give unique names to all the tensors
	NodeNames names = NodeNames();
	map<int, int> cluster_var_index = map<int, int>();
    int mem_index = 0;
	for (auto node = ir.begin(); !node.is_end(); ++node) {
		if (node->name == "memory")
		{
			names[*node] = "mem" + to_string(mem_index);
			mem_index++;
		}
		else
		{
			int cluster_id = node->cluster_id_;
			int var_index = cluster_var_index[cluster_id];
			names[*node] = "var" + to_string(cluster_id) + "_" + to_string(var_index);
			cluster_var_index[cluster_id]++;
		}
	}

	// now create the listing
	string listing;
	int indent = 0;
	int prev_cluster_id = -1;
	for (auto node = ir.begin(); !node.is_end(); ++node) {
		if (compact) {
			if (node->name == "const") continue;
		}

		if (node->name == "loop_end") {
			indent--;
		}

		if (node->cluster_id_ != -1) {
			if (node->cluster_id_ != prev_cluster_id) {
				listing += "\n";
			}
			listing += to_string(node->cluster_id_) + ": ";
		}

		// indent
		for (int i = 0; i < indent; i++) {
			listing += "  ";
		}

		if (node->tensor_->type != DataType::None) {
			// print the tensor name
			listing += names[*node] + " = ";
		}

		listing += node->name + "(";

		Arguments inputs = node->GetArguments(Argument::Type::Input);
		Arguments indices = node->GetArguments(Argument::Type::Index);
		Arguments shape = node->GetArguments(Argument::Type::Shape);
		Arguments memory = node->GetArguments(Argument::Type::Memory);

		if (!memory.empty()) {
			listing += "memory=[";
			for (int i = 0; i < memory.size(); i++) {
				if (i != 0) listing += ",";
				listing += GetNodeName(memory[i].from_->get(), names, compact);
			}
			listing += "], ";
		}

		if (!inputs.empty()) {
			listing += "inputs=[";
			for (int i = 0; i < inputs.size(); i++) {
				if (i != 0) listing += ",";
				listing += GetNodeName(inputs[i].from_->get(), names, compact);
			}
			listing += "], ";
		}

		if (!indices.empty()) {
			listing += "indices=[";
			for (int i = 0; i < indices.size(); i++) {
				if (i != 0) listing += ",";
				listing += GetNodeName(indices[i].from_->get(), names, compact);
			}
			listing += "], ";
		}

		if (!shape.empty()) {
			listing += "shape=[";
			for (int i = 0; i < shape.size(); i++) {
				if (i != 0) listing += ",";
				listing += GetNodeName(shape[i].from_->get(), names, compact);
			}
			listing += "], ";
		}

		if (!node->tensor_->data.empty()) {
			listing += "data=[";
			for (int i = 0; i < node->tensor_->data.size(); i++) {
				if (i != 0) listing += ",";
				listing += to_string(node->tensor_->data[i]);
			}
			listing += "], ";
		}
		
		switch (node->memory_type_) {
			case MemoryType::Input:
				listing += "memory_type=input, ";
				break;
			case MemoryType::Output:
				listing += "memory_type=output, ";
				break;
			case MemoryType::Constant:
				listing += "memory_type=constant, ";
				break;
			default:
				break;
		}	

		if (node->tensor_->type != DataType::None) {
			listing += "type = " + DataTypeToString(node->tensor_->type);
		}

		listing += ")\n";

		if (node->name == "loop_begin") {
			indent++;
		}

		prev_cluster_id = node->cluster_id_;
	}

	return listing;
}

}