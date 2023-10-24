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

string GetOperationListing(const IR& ir, bool compact) {
	
    list<Tensor*> nodes = ir.GetNodes();

	// first give unique names to all the tensors
	TensorNames names = TensorNames();
	int index = 0;
	for (const Tensor* node : nodes) {
		names[node] = "t" + to_string(index);
		index++;
	}

	// now create the listing
	string listing;
	int indent = 0;
	for (const Tensor* node : nodes) {
		if (compact) {
			if (node->name == "const") continue;
		}

		if (node->name == "loop_end") {
			indent--;
		}

		// indent
		for (int i = 0; i < indent; i++) {
			listing += "  ";
		}

		if (node->type != DataType::None) {
			// print the tensor name
			listing += names[node] + " = ";
		}

		listing += node->name + "(";

		Arguments inputs = node->GetArguments(Argument::Type::Input);
		Arguments indices = node->GetArguments(Argument::Type::Index);
		Arguments shape = node->GetArguments(Argument::Type::Shape);

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

		if (!node->data.empty()) {
			listing += "data=[";
			for (int i = 0; i < node->data.size(); i++) {
				if (i != 0) listing += ",";
				listing += to_string(node->data[i]);
			}
			listing += "], ";
		}

		if (node->type != DataType::None) {
			listing += "type = " + DataTypeToString(node->type);
		}

		listing += ")\n";

		if (node->name == "loop_begin") {
			indent++;
		}
	}

	return listing;
}

}