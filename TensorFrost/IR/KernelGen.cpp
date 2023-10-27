#pragma once

#include "IR/KernelGen.h"

namespace TensorFrost {

bool IsBoundary(const Node* input, const Node* output, int arg_index,
                Argument::Type arg_type, bool is_identity) {

	if (arg_index >= 0) 
	{
		OpType input_type = input->op->GetOpType();
		OpType output_type = output->op->GetOpType();

		bool isFromScatter = input_type == OpType::Scatter;
		bool isToScatter = output_type == OpType::Scatter;
		bool isFromStore = input_type == OpType::Store;
		bool isFromOutput = input->memory_type_ == MemoryType::Output;

		if (isFromScatter || isFromStore) {
			if (isFromScatter && isToScatter) {
				return false;  // multiple scatters can be merged
			} else {
				return true;
			}
		}

		if (output_type == OpType::Load || output_type == OpType::Store) {
			return arg_index == 0 && arg_type == Argument::Type::Input && !is_identity;
		}

		if (output_type == OpType::Scatter) {
			return arg_index == 0 && arg_type == Argument::Type::Input;
		}
	}

	// TODO write shape comparison function
	// if(input.tensor_.shape_ != output.tensor_.shape_)
	//{
	//     if(!isToScatter)
	//     {
	//         return true;
	//     }
	// }

	// memory should not be inside work clusters
	if ((input->name == "memory" || output->name == "memory") && (input->name != output->name)) {
		return true;
	}

	return false;
}

void IR::Clusterize() {
	int cluster_id = 0;
	for (auto node = begin(); !node.is_end(); ++node) {
		// check if node is a cluster edge
		Tensor* tensor = node->tensor_;
		
		Arguments indices = node->GetArguments(Argument::Type::Index);
		bool identity = indices.size() == 0;

		bool is_boundary = false;
		Node* prev = node.get_prev();
		if (prev != nullptr) {
			if (prev->cluster_id_ == cluster_id &&
			    IsBoundary(prev, *node, -1, Argument::Type::None, identity)) {
				is_boundary = true;
			}
		}

		// go over all inputs
		for (auto& input : tensor->node->inputs_) {
			// check if input is the boundary of this cluster
			if (input.from_->get()->cluster_id_ == cluster_id &&
			    IsBoundary(input.from_->get(), *node, input.index_, input.type_, identity)) {
				is_boundary = true;
				break;
			}
		}

		if (is_boundary) {
			cluster_id++;
		}

		node->cluster_id_ = cluster_id;
	}
}

void IR::PostProcessClusters() {
	//get cluster ranges
	map<int, vector<Node*>> cluster_outputs;
	map<int, Node*> cluster_begin;
	UpdateNodeOutputs();

	//find all nodes that are outputs of a cluster (i.e. point to a node outside the cluster)
	for (auto node = begin(); !node.is_end(); ++node) {
		if (node->name == "memory") {
			continue;
		}

		Node* prev = node.get_prev();
		if (prev == nullptr || (prev->cluster_id_ != node->cluster_id_)) {
			cluster_begin[node->cluster_id_] = *node;
		}

		bool is_output = node->memory_type_ == MemoryType::Output;
		for (auto& output : node->outputs_) {
			Node* output_node = output->to_->get();
			if (output_node->cluster_id_ != node->cluster_id_) {
				is_output = true;
				break;
			}
		}

		if (is_output)
		{
			cluster_outputs[node->cluster_id_].push_back(*node);
		}
	}

	//go over all outputs of each cluster and create memory nodes to store the output
	for (auto cluster_out : cluster_outputs)
	{
		vector<Node*> cluster_outs = cluster_out.second;
		int cluster_id = cluster_out.first;

		for (auto output : cluster_outs)
		{
			Node* mem;
			// add memory node before this cluster
			ExecuteExpressionBefore(cluster_begin[cluster_id],[&]() 
			{
				mem = Tensor::Memory(output->GetArguments(Argument::Type::Shape),
										output->tensor_->type).node;

				if (output->memory_type_ == MemoryType::Output) {
					mem->memory_type_ = MemoryType::Output;
					output->memory_type_ = MemoryType::None;
				}
			});

			// all the nodes must now use to the memory node
			// which stores the output result not the output node itslef
			SwapLables(output, mem);

			// add store node after this node
			ExecuteExpressionAfter(output, [&]()
			{
				// add store node after this node
				Tensor::Store(*mem->tensor_, *output->tensor_);
				cursor_->cluster_id_ = cluster_id;
			});
		}
	}


	//go over all nodes, and add load nodes (if not already) for all inputs that are not in the cluster
	for (auto node = begin(); !node.is_end(); ++node) {
		if (node->op->GetOpType() == OpType::Memory ||
			node->op->GetOpType() == OpType::Store ||
			node->op->GetOpType() == OpType::Load || 
			node->op->GetOpType() == OpType::Scatter) {
			continue;
		}

		// go over all inputs
		for (auto& input : node->inputs_) {
			// check if input is the boundary of this cluster
			if (input.from_->get()->cluster_id_ != node->cluster_id_ && input.type_ != Argument::Type::Shape) {
				// add load node before this node
				ExecuteExpressionBefore(*node, [&]()
				{
					//get memory node
					Tensor* mem = input.from_->get()->tensor_;
					Tensor& loaded = Tensor::Load(*mem);
					loaded.node->cluster_id_ = node->cluster_id_;
					// the node must now use the loaded node
					input.from_ = loaded.node->GetLable();
				});
			}
		}
	}	
}

}  // namespace TensorFrost