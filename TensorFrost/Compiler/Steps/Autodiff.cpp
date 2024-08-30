#include "Compiler/KernelGen.h"

namespace TensorFrost {

void ComputeNodeGradients(Node* value, const Tensor* grad, NodeGrads& grads)
{
	try {
		string op_name = value->name;
		//add input arguments
		if(value->flags.has(NodeProp::PassGrad)) {
			op_name = "passthrough_grad";
		}
		if(value->flags.has(NodeProp::DetachGrad)) {
			op_name = "detached_grad";
		}

		VJPGradientFunction gradient_func = GetVJPForOperation(op_name);

		Tensor out = *value->tensor_;
		gradient_func(value->args, out, *grad, grads);
	} catch (const std::exception& e) {
		throw std::runtime_error("Error in gradient computation for " + value->debug_name + "(" + to_string(value->debug_index) + "): " + e.what());
	}
}

void IR::ComputeAutodiff()
{
	vector<Node*> gradients = GetNodesOfType(OpProp::Gradient);

	if(gradients.empty()) {
		return;
	}

	set<Node*> loss_nodes;
	map<pair<Node*, Node*>, Node*> loss_wrt_grad;
	unordered_map<Node*, int> min_range; //index of earliest node required for the gradient, end of backpropagation

	for (auto gradient : gradients) {
		Node* loss = gradient->args.Get(ArgType::Input, 0);
		Node* wrt = gradient->args.Get(ArgType::Input, 1);
		Node* last_loss_version = loss->GetLastVersion(gradient);

		loss_nodes.insert(last_loss_version);
		min_range[last_loss_version] = std::min(min_range[last_loss_version], wrt->index_);
		loss_wrt_grad[{last_loss_version, wrt}] = gradient;
	}

	map<Node*, Node*> grad_to_computed_grad;
	for (auto loss : loss_nodes) {
		set<Node*> visited;
		map<Node*, const Tensor*> node_to_grad;

		unordered_set<Node*> loss_deps = GetDependencies({loss});

		//get all differentiable nodes that can change the loss
		vector<Node*> queue;
		for (auto dep : loss_deps) {
			bool in_range = (dep->index_ <= loss->index_ && dep->index_ >= min_range[loss]);
			bool dep_is_accessible = dep->HasCommonParents(loss); //is it in scope of the loss
			if(in_range && !dep->op->HasAllTypes(OpProp::Nondiff) &&
			   dep_is_accessible && (dep->type == TFType::Float || dep->op->HasAllTypes(OpProp::Modifier))) {
				queue.push_back(dep);
			}
		}

		//sort the nodes by index in descending order (backpropagation)
		ranges::sort(queue.begin(), queue.end(), [](Node* a, Node* b) {
			return a->index_ > b->index_;
		});

		Node* loss_value = loss;
		if(loss->op->HasAllTypes(OpProp::Modifier)) {
			loss_value = loss->args.Get(ArgType::Memory);
		}

		ExecuteExpressionAfter(loss, [&]() {
			node_to_grad[loss_value] = &Tensor::Constant(1.0f);
			for(auto node : queue) {
				if(!node_to_grad.contains(node) && !node->op->HasAllTypes(OpProp::Modifier)) {
					continue;
				}

				node_to_grad[node] = &ReduceGradientToShape(*node_to_grad[node], *node->GetTensor());

				NodeGrads grads = NodeGrads(node, node_to_grad);
				ComputeNodeGradients(node, node_to_grad[node], grads);

				//store the computed gradients
				for (auto& [id, input]: node->args.inputs_) {
					if(!grads.Contains(id)) {
						continue;
					}

					const Tensor& new_grad = *grads.GetGrad(id);
					node_to_grad[input] = &new_grad;

					//TODO: maybe add a function to get temp names
					if(input->debug_name != "") {
						new_grad.SetDebugName("d" + loss_value->debug_name + "_d" + input->debug_name);
					} else if(input->var_name != "") {
						new_grad.SetDebugName("d" + loss_value->debug_name + "_d" + input->var_name);
					}
				}
			}
		});

		for (auto wrt_grad : loss_wrt_grad) {
			if (wrt_grad.first.first != loss) {
				continue;
			}

			Node* grad = wrt_grad.second;
			if(!node_to_grad.contains(wrt_grad.first.second)) {
				throw std::runtime_error("Gradient not computed for " + wrt_grad.first.second->var_name);
			}
			Node* computed_grad = node_to_grad[wrt_grad.first.second]->node_;
			grad_to_computed_grad[grad] = computed_grad;
		}

		UpdateGraph();
	}

	unordered_set<Node*> nodes_to_remove;
	//replace all gradients with computed gradients
	for (auto gradient : gradients) {
		Node* computed_grad = grad_to_computed_grad[gradient];
		//replace the node with the sum
		gradient->ReplaceThisWithGivenNode(computed_grad);

		//mark the node for removal
		nodes_to_remove.insert(gradient);
		UpdateGraph();
	}

	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();
}

} // namespace TensorFrost
