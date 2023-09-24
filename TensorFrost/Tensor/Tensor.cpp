#include "Tensor.h"

namespace TensorFrost {
IR* Tensor::graph_ = nullptr;

using TensorNames = std::unordered_map<const Tensor *, string>;

string GetNodeName(const Tensor* tensor, TensorNames& names) {
	if (tensor->name == "const") {
		return tensor->GetConstantString();
	} 
		return names[tensor];

}

string IR::GetOperationListing() {
	// return to_string(nodes.size()) + " operations\n";

	// first give unique names to all the tensors
	TensorNames names = TensorNames();
	int index = 0;
	for (const shared_ptr<Tensor>& node : nodes_) {
		names[node.get()] = "t" + to_string(index);
		index++;
	}

	// now create the listing
	string listing;
	for (const shared_ptr<Tensor>& node : nodes_) {
		if (node->name == "const") continue;

		listing += names[node.get()];

		listing += " = " + node->name + "(";

		Arguments inputs = node->GetArguments(Argument::Type::Input);
		Arguments indices = node->GetArguments(Argument::Type::Index);
		Arguments shape = node->GetArguments(Argument::Type::Shape);

		if (!inputs.empty()) {
			listing += "inputs=[";
			for (int i = 0; i < inputs.size(); i++) {
				if (i != 0) listing += ",";
				listing += GetNodeName(inputs[i].tensor, names);
			}
			listing += "] ";
		}

		if (!indices.empty()) {
			listing += "indices=[";
			for (int i = 0; i < indices.size(); i++) {
				if (i != 0) listing += ",";
				listing += GetNodeName(indices[i].tensor, names);
			}
			listing += "] ";
		}

		if (!shape.empty()) {
			listing += "shape=[";
			for (int i = 0; i < shape.size(); i++) {
				if (i != 0) listing += ",";
				listing += GetNodeName(shape[i].tensor, names);
			}
			listing += "] ";
		}

		if (!node->data.empty()) {
			listing += "data=[";
			for (int i = 0; i < node->data.size(); i++) {
				if (i != 0) listing += ",";
				listing += to_string(node->data[i]);
			}
			listing += "] ";
		}

		listing += ")\n";
	}

	return listing;
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

}  // namespace TensorFrost