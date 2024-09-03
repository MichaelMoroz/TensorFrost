#include "Compiler/KernelGen.h"

namespace TensorFrost {
bool IR::InsertAlgorithmicPrimitives() {
	// get all nodes for each type
	vector<Node*> nodes = GetNodesOfType(OpProp::Algorithm);

	unordered_set<Node*> nodes_to_remove;

	// replace all nodes with the algorithmic primitive
	for (auto node : nodes) {
		//compute the sum after the node
		ExecuteExpressionAfter(node, [&]() {
			//get the input tensor
			map<int, const Tensor*> inputs = node->args.GetTensors(ArgType::Input);

			//get sum axis
			vector<int> axes;
			for (int i = 0; i < node->data.size(); i++) {
				axes.push_back((int)node->data[i]);
			}

			Tensors results;
			ImplementationFunction func = GetImplementationForOperation(node->name);
			func(results, inputs, node->GetTensor(), axes);

			const Tensor* result = results[0];

			//replace the node with the sum
			node->ReplaceThisWithGivenNode(result->node_);

			ShapeCompareResult shape_result = CompareShape(node, result->node_, true);
			if (!shape_result.compatible) {
				throw std::runtime_error("Algorithmic primitive " + node->name + " at " + node->debug_name + " has incompatible shapes");
			}
		});

		//mark the node for removal
		nodes_to_remove.insert(node);

		UpdateGraph();
	}

	// remove all nodes that are not used
	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();

	return nodes_to_remove.empty();
}

} // namespace TensorFrost
