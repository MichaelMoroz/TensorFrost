#include "Compiler/KernelGen.h"

namespace TensorFrost {
bool IR::InsertAlgorithmicPrimitives(bool skip_differentiable) {
	// get all nodes for each type
	vector<Node*> nodes = GetNodesOfType(OpProp::Algorithm);

	unordered_set<Node*> nodes_to_remove;

	// replace all nodes with the algorithmic primitive
	for (auto node : nodes) {
		if(HasDerivativeImplemented(node->name) && skip_differentiable) {
			continue;
		}
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

#ifndef NDEBUG
			current_function = node->name;
#endif

			func(results, inputs, node->GetTensor(), axes);

#ifndef NDEBUG
			current_function = "None";
#endif

			const Tensor* result = results[0];

			//replace the node with the sum
			node->ReplaceThisWithGivenNode(result->node_);

			ShapeCompareResult shape_result = CompareShape(node, result->node_);
			if (!shape_result.exactly_compatible) {
				throw std::runtime_error("Algorithmic primitive " + node->name + " at " + node->debug_name + " has incompatible shapes");
			}
		});

		//mark the node for removal
		nodes_to_remove.insert(node);
	}

	// remove all nodes that are not used
	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();

	return nodes_to_remove.empty();
}

} // namespace TensorFrost
