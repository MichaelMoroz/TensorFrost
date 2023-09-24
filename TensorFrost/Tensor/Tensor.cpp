#include "Tensor.h"

namespace TensorFrost {
IR* Tensor::graph_ = nullptr;

typedef std::unordered_map<const Tensor*, string> TensorNames;

string GetNodeName(const Tensor* tensor, TensorNames& names) {
	if(tensor->name == "const") {
		return tensor->GetConstantString();
	} else {
		return names[tensor];
	}
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
		vector<int> shape = node->TryGetShape();

		listing += names[node.get()] + "(";

		for (int i = 1; i < shape.size(); i++) {
			if (i != 1) listing += ",";
			listing += to_string(shape[i]);
		}
		
		listing += ") = " + node->name + "(" + node->GetConstantString();

		for (int i = 0; i < node->inputs.size(); i++) {
			if (i != 0) listing += ",";
			listing += GetNodeName(node->inputs[i].tensor, names);
		}

		listing += ")\n";
	}

	return listing;
}
}  // namespace TensorFrost