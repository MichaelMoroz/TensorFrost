#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

string GetOperationListing(const IR& ir, bool compact) {
	// first give unique names to all the tensors
	NodeNames names = GenerateNodeNames(ir);
	ClusterProp clusters = ir.GetClusterProperties();

	// now create the listing
	string listing;
	int indent = 0;
	Lable* prev_cluster_head = nullptr;
	for (auto node = ir.begin(); !node.is_end(); ++node) {
		if (compact) {
			if (node->name == "const") continue;
		}

		if (node->name == "loop_end") {
			indent--;
		}

		if (node->cluster_head_ != prev_cluster_head) {
			listing += "\n";
		}

		if (!compact && node->cluster_head_ != nullptr) {
			listing += GetNodeName(node->cluster_head_->node_, names, compact) + ": ";
		}

		// indent
		for (int i = 0; i < indent; i++) {
			listing += "  ";
		}

		if (node->tensor_->type != DataType::None) {
			// 
			//  the tensor name
			listing += names[*node] + " = ";
		}

		listing += node->name + "(";

		Arguments inputs = node->GetArguments(Arg::Type::Input);
		Arguments indices = node->GetArguments(Arg::Type::Index);
		Arguments shape = node->GetArguments(Arg::Type::Shape);
		Arguments memory = node->GetArguments(Arg::Type::Memory);

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

		if (clusters.node_cost[node.get()] != 0) {
			listing += "cost=" + to_string(clusters.node_cost[node.get()]) + ", ";
		}

		if (node->tensor_->type != DataType::None) {
			listing += "type=" + DataTypeToString(node->tensor_->type);
		}

		listing += ")\n";

		if (node->name == "loop_begin") {
			indent++;
		}

		prev_cluster_head = node->cluster_head_;
	}

	return listing;
}

}  // namespace TensorFrost