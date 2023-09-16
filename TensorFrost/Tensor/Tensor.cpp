#include "Tensor.h"

namespace TensorFrost {
IR* Tensor::graph_ = nullptr;

string IR::GetOperationListing() {
	// return to_string(nodes.size()) + " operations\n";

	// first give unique names to all the tensors
	std::unordered_map<Tensor*, string> names =
	    std::unordered_map<Tensor*, string>();
	int index = 0;
	for (const shared_ptr<Tensor>& node : nodes_) {
		names[node.get()] = "t" + to_string(index);
		index++;
	}

	// now create the listing
	string listing;
	for (const shared_ptr<Tensor>& node : nodes_) {
		listing += names[node.get()] + "(" + to_string(node->shape[0]);
		for (int i = 1; i < node->shape.size(); i++) {
			listing += "x" + to_string(node->shape[i]);
		}
		listing += ") = " + node->name + "(" + node->GetConstantString();
		for (int i = 0; i < node->inputs.size(); i++) {
			listing += names[node->inputs[i].tensor];
			if (i < node->inputs.size() - 1) {
				listing += ", ";
			}
		}
		listing += ")\n";
	}

	return listing;
}
}  // namespace TensorFrost