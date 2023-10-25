#pragma once

#include "IR/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

string GetNodeName(const Tensor* tensor, TensorNames& names, bool compact) {
	if (compact)
	{
		if (tensor->name == "const") {
			return tensor->GetConstantString();
		}
	}
	return names[tensor];
}

inline string Tensor::GetConstantString() const {
	if (name == "const" || name == "dim_id") {
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
	list<const Node*> nodes = ir.GetNodes();

	// first give unique names to all the tensors
	TensorNames names = TensorNames();
	int index = 0;
	for (const Node* node : nodes) {
		names[node->tensor_] = "t" + to_string(index);
		index++;
	}

	// now create the listing
	string listing;
	int indent = 0;
	for (const Node* node : nodes) {
		if (compact) {
			if (node->tensor_->name == "const") continue;
		}

		if (node->tensor_->name == "loop_end") {
			indent--;
		}

		// indent
		for (int i = 0; i < indent; i++) {
			listing += "  ";
		}

		if (node->tensor_->type != DataType::None) {
			// print the tensor name
			listing += names[node->tensor_] + " = ";
		}

		listing += node->tensor_->name + "(";

		Arguments inputs = node->tensor_->GetArguments(Argument::Type::Input);
		Arguments indices = node->tensor_->GetArguments(Argument::Type::Index);
		Arguments shape = node->tensor_->GetArguments(Argument::Type::Shape);

		if (!inputs.empty()) {
			listing += "inputs=[";
			for (int i = 0; i < inputs.size(); i++) {
				if (i != 0) listing += ",";
				listing += GetNodeName(inputs[i].tensor, names, compact);
			}
			listing += "], ";
		}

		if (!indices.empty()) {
			listing += "indices=[";
			for (int i = 0; i < indices.size(); i++) {
				if (i != 0) listing += ",";
				listing += GetNodeName(indices[i].tensor, names, compact);
			}
			listing += "], ";
		}

		if (!shape.empty()) {
			listing += "shape=[";
			for (int i = 0; i < shape.size(); i++) {
				if (i != 0) listing += ",";
				listing += GetNodeName(shape[i].tensor, names, compact);
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

		if (node->cluster_id_ != -1)
		{
			listing += "cluster_id = " + to_string(node->cluster_id_) + ", ";
		}

		if (node->tensor_->type != DataType::None) {
			listing += "type = " + DataTypeToString(node->tensor_->type);
		}

		listing += ")\n";

		if (node->tensor_->name == "loop_begin") {
			indent++;
		}
	}

	return listing;
}

}