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
			return arg_type == Argument::Type::Memory && !is_identity;
		}

		if (output_type == OpType::Scatter) {
			return arg_type == Argument::Type::Memory;
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

map<Node*, Node*> IR::CopyComputation(unordered_set<Node*> targets) {
	// do a depth first search to copy all the nodes required for the targets
	unordered_set<Node*> nodes_to_copy;
	std::function<void(Node*)> dfs = [&](Node* node) {
		if (nodes_to_copy.find(node) != nodes_to_copy.end()) {
			return;
		}
		nodes_to_copy.insert(node);
		for (auto& input : node->inputs_) {
			if (input.type_ == Argument::Type::Memory || input.type_ == Argument::Type::Shape)
				continue;

			dfs(input.from_->get());
		}
	};

	for (Node* target : targets) {
		dfs(target);
	}

	// copy the nodes
	map<Node*, Node*> copied_node_map;
	for (auto node = begin(); !node.is_end(); ++node) {
		if (!nodes_to_copy.contains(node.get())) {
			continue;
		}

		// create new arguments
		Arguments new_args;
		for (Argument& arg : node->inputs_) {
			//if shape or memory argument, then no need to use copied node
			if (arg.type_ == Argument::Type::Memory || arg.type_ == Argument::Type::Shape) {
				new_args.push_back(arg);
				continue;
			}

			Node* from = arg.from_->get();

			if (!copied_node_map.contains(from)) {
				throw std::runtime_error("Node not found");
			}

			// create new argument
			new_args.push_back(
			    Argument(arg.type_, copied_node_map[from]->GetLable(), arg.index_));
		}

		// create new node
		Tensor* tensor = Tensor::GetCopy(*node->tensor_, new_args);
		Node* new_node = tensor->node;
		copied_node_map[node.get()] = new_node;
	}

	return copied_node_map;
}

void IR::OptimizeClusters()
{
	//get cluster data
	ClusterProp clusters = GetClusterProperties();

	//go over each cluster and copy computations outside the cluster if they are cheap enough
	for (auto cluster_begin : clusters.cluster_heads)
	{
		Node* begin = cluster_begin->node_;

		if (begin->name == "memory") {
			continue;
		}

		unordered_set<Argument*> args_to_copy;
		//go over all nodes in the cluster and check if their inputs can be copied
		for (auto node = iterator(begin); !node.is_cluster_end(cluster_begin);
		     ++node) {
			//go over all inputs
			for (auto& input : node->inputs_) {
				//if input is memory or shape, then skip
				if (input.type_ == Argument::Type::Memory || input.type_ == Argument::Type::Shape || input.from_->get()->name == "memory")
				{
					continue;
				}

				//if input is outside the cluster, then it can be copied
				if (input.from_->get()->cluster_head_ != node->cluster_head_) {
					//check if input is cheap enough to copy
					if (clusters.node_cost[input.from_->get()] < 256.0f) {
						args_to_copy.insert(&input);
					}
				}
			}
		}

		unordered_set<Node*> nodes_to_copy;
		for (auto arg : args_to_copy) {
			nodes_to_copy.insert(arg->from_->get());
		}

		//copy all the nodes at the beginning of the cluster
		map<Node*, Node*> copied_node_map;
		ExecuteExpressionBefore(begin, [&]() { copied_node_map = CopyComputation(nodes_to_copy); });

		//replace all the arguments that use the copied nodes
		for (auto arg : args_to_copy) {
			arg->from_ = copied_node_map[arg->from_->get()]->GetLable();
		}
	}
}

void IR::RemoveUnusedNodes()
{
	//use depth first search to find all nodes that are used for the output nodes
	unordered_set<Node*> used_nodes;
	
	//go over all inputs of the output nodes
	std::function<void(Node*)> dfs = [&](Node* node) {
		if (used_nodes.contains(node)) {
			return;
		}
		used_nodes.insert(node);
		for (auto& input : node->inputs_) {
			dfs(input.from_->get());
		}
	};

	for (auto node = begin(); !node.is_end(); ++node) {
		if (node->memory_type_ == MemoryType::Output) {
			dfs(node.get());
		}
	}

	//remove all nodes that are not used
	unordered_set<Node*> nodes_to_remove;
	for (auto node = begin(); !node.is_end(); ++node) {
		if (node->name == "memory") {
			continue;
		}
		if (!used_nodes.contains(node.get())) {
			nodes_to_remove.insert(node.get());
		}
	}

	for (auto node : nodes_to_remove) {
		RemoveNode(node);
	}
}

void IR::Clusterize() {
	Lable* current_cluster = nullptr;
	for (auto node = begin(); !node.is_end(); ++node) {
		// remove old cluster head
		if (node->cluster_head_ != nullptr) {
			//if last cluster node, delete lable
			if (node->next_ == nullptr || node->next_->cluster_head_ != node->cluster_head_) {
				delete node->cluster_head_;
			}
			node->cluster_head_ = nullptr;
		}

		// check if node is a cluster edge
		Tensor* tensor = node->tensor_;
		
		Arguments indices = node->GetArguments(Argument::Type::Index);
		bool identity = indices.size() == 0;

		bool is_boundary = false;
		Node* prev = node.get_prev();
		if (prev != nullptr) {
			if (prev->cluster_head_ == current_cluster &&
			    IsBoundary(prev, *node, -1, Argument::Type::None, identity)) {
				is_boundary = true;
			}
		}

		// go over all inputs
		for (auto& input : tensor->node->inputs_) {
			// check if input is the boundary of this cluster
			if (input.from_->get()->cluster_head_ == current_cluster &&
			    IsBoundary(input.from_->get(), *node, input.index_, input.type_, identity)) {
				is_boundary = true;
				break;
			}
		}

		if (is_boundary) {
			current_cluster = new Lable(*node);
		}

		node->cluster_head_ = current_cluster;
	}
}

ClusterProp IR::GetClusterProperties() const
{
    map<Lable*, vector<Node*>> cluster_outputs;
	map<Node*, vector<Argument*>> node_output;
	map<Node*, float> node_cost;
	unordered_set<Lable*> cluster_heads;

	UpdateNodeOutputs();
	//find all nodes that are outputs of a cluster (i.e. point to a node outside the cluster)
	for (auto node = begin(); !node.is_end(); ++node) {
		if (node->name == "memory" || node->cluster_head_ == nullptr) {
			continue;
		}

		float input_cost = node->op->GetCost();
		for (auto& input : node->inputs_) {
			if (input.from_->get()->cluster_head_ == node->cluster_head_ &&
			    input.type_ != Argument::Type::Memory && input.type_ != Argument::Type::Shape) {
				input_cost += node_cost[input.from_->get()];
			}
		}
		node_cost[*node] = input_cost;

		bool is_output = node->memory_type_ == MemoryType::Output;

		vector<Argument*> outputs;
		for (auto& output : node->outputs_) {
			if (output->to_ == nullptr) continue;
			//if is a shape or memory argument, then skip (shape is loaded on CPU)
			if (output->type_ == Argument::Type::Shape) {
				continue;
			}
			Node* output_node = output->to_->get();
			if (output_node->cluster_head_ != node->cluster_head_) {
				outputs.push_back(output);
				is_output = true;
			}
		}

		if (is_output)
		{
			cluster_outputs[node->cluster_head_].push_back(*node);
			node_output[*node] = outputs;
		}

		cluster_heads.insert(node->cluster_head_);
	}

    return ClusterProp(cluster_outputs, node_output, node_cost, cluster_heads);
}

void IR::PostProcessClusters() {
	//get cluster data
	ClusterProp clusters = GetClusterProperties();

	//go over all outputs of each cluster and create memory nodes to store the output
	for (auto cluster_out : clusters.output)
	{
		vector<Node*> cluster_outs = cluster_out.second;
		Lable* cluster_head = cluster_out.first;

		for (auto output : cluster_outs)
		{
			Node* mem;
			// add memory node before this cluster
			ExecuteExpressionBefore(cluster_head->node_,[&]() 
			{
				mem = Tensor::Memory(output->GetArguments(Argument::Type::Shape),
										output->tensor_->type).node;

				if (output->memory_type_ == MemoryType::Output) {
					mem->memory_type_ = MemoryType::Output;
					output->memory_type_ = MemoryType::None;
				}
			}, false);

			// go over all outputs of this node and replace their input with the memory node
			for (auto& arg_out : clusters.node_output[output])
			{
				if(arg_out->type_ != Argument::Type::Shape && arg_out->type_ != Argument::Type::Memory)
				{	
					// if not a memory or shape argument, then the memory needs to be loaded before the node
					ExecuteExpressionBefore(arg_out->to_->get(), [&]()
					{
						Tensor& loaded = Tensor::Load(*mem->tensor_);
						//loaded.node->cluster_id_ = arg_out->to_->get()->cluster_id_;
						// the node must now use the loaded value
						arg_out->from_ = loaded.node->GetLable();
					});
				}
				else
				{
					// otherwise the memory can be used directly
					arg_out->from_ = mem->GetLable();
				}
			}

			// add store node after this node
			ExecuteExpressionAfter(output, [&]()
			{
				// add store node after this node
				Tensor* store = &Tensor::Store(*mem->tensor_, *output->tensor_);
				store->node->cluster_head_ = cluster_head;
			});
		}
	}


	//replace all inputs pointing to memory nodes with the memory node
	for(auto node = begin(); !node.is_end(); ++node)
	{
		if(node->name == "memory") continue;

		for(auto& input : node->inputs_)
		{
			if (input.type_ == Argument::Type::Memory || input.type_ == Argument::Type::Shape)
				continue;

			if(input.from_->get()->name == "memory")
			{
				//load the memory node before this node
				ExecuteExpressionBefore(node.get(), [&]()
				{
					Tensor& loaded = Tensor::Load(*input.from_->get()->tensor_);
					input.from_ = loaded.node->GetLable();
				});
			}
		}
	}
}

void IR::TransformToLinearIndex()
{
	ClusterProp clusters = GetClusterProperties();

	//replace all dim_id nodes with the corresponding index computed from thread_id
	for (auto cluster_begin : clusters.cluster_heads)
	{
		Node* begin = cluster_begin->node_;

		// add thread node
		Tensor* thread_index;
		ExecuteExpressionBefore(begin, [&]()
		{ 
			thread_index = &begin->tensor_->ThreadIndex();
		});

		// load kernel shape
		map<int, Tensor*> kernel_shape_map = begin->GetArgumentTensors(Argument::Type::Shape);
		Tensors kernel_shape;
		for (auto& shape : kernel_shape_map) {
			kernel_shape.push_back(shape.second);
		}

		if (kernel_shape.size() == 0)
		{
			//can skip if no kernel shape - no index
			continue;
		}

		// compute the index for each dimension
		int dims = kernel_shape.size();
		Tensors indices = Tensors(dims);
		ExecuteExpressionBefore(begin, [&]()
		{
			for (int i = 0; i < dims; i++) {
				Tensor* div = &Tensor::Constant(kernel_shape, 1);
				
				for (int j = i + 1; j < dims; j++) {
					div = &(*div * *kernel_shape[dims - j - 1]);
				}

				Tensor& dim = *thread_index / *div;

				if (i > 0) {
					dim = dim % *kernel_shape[dims - i - 1];
				}

				indices[dims - i - 1] = &dim;
			}
		});

		// replace all dim nodes with the corresponding index node
		unordered_set<Node*> nodes_to_remove;
		for (auto node = iterator(begin); !node.is_cluster_end(cluster_begin);
		     ++node) {
			if (node->name == "dim_id") {
				int dim = node->tensor_->data[0];
				if (dim >= dims) {
					throw runtime_error("Invalid dimension index " + to_string(dim) +
										                    " for kernel of size " + to_string(dims));
				}

				//swap the dim node with the corresponding index node
				CopyLable(node.get(), indices[dim]->node);

				//remove the dim node
				nodes_to_remove.insert(node.get());
			}
		}

		 // go over all nodes that take an index as input (e.g. load, store, atomic)
		for (auto node = iterator(begin); !node.is_cluster_end(cluster_begin);
		     ++node) {
			auto op_type = node->op->GetOpType();
			if ( op_type == OpType::Load || op_type == OpType::Store || op_type == OpType::Scatter) {
				ExecuteExpressionBefore(*node, [&]()
				{
					// get the input memory node
					const Tensor* memory =
					    node.get()->GetArgumentTensors(Argument::Type::Memory)[0];
					int memory_dim = kernel_shape.size(); //TODO get memory dim from memory tensor instead of kernel shape

					// get the index nodes
					map<int, Tensor*> idx = node->GetArgumentTensors(Argument::Type::Index);

					//function to get index for given dimension, if not found then return default dim index
					std::function<Tensor*(int)> get_index = [&](int dim) {
						if (idx.find(dim) != idx.end()) {
							return idx[dim];
						} else {
							return const_cast<Tensor*> (indices[dim]);
						}
					};

					// compute the flat index
					Tensor* flat_index = get_index(memory_dim - 1);
					for (int i = memory_dim - 2; i >= 0; i--) {
						*flat_index = *flat_index * *kernel_shape[i];
						*flat_index = *flat_index + *get_index(i);
					}

					// TODO clamp each dimension index individually instead of the flat
					// index
					// TODO add different modes for clamping (e.g. clamp, wrap, mirror,
					// zero)

					// remove the index node edges
					node->RemoveArguments(Argument::Type::Index);

					// add the flat index node edge
					node->AddArgument(flat_index->node, Argument::Type::Index, 0);
				});
			}
		}

		// remove all dim nodes
		for (auto node : nodes_to_remove) {
			RemoveNode(node);
		}
	}
}

Program* GenerateProgram(IR* ir) {
	Program* program = new Program(ir);

	// go over all clusters, find their type, and add them to the program if they
	// are used
	for (auto node = ir->begin(); !node.is_end(); ++node) {
		Node* begin;

		if (node.is_cluster_begin()) {
			begin = node.get();
		} else {
			continue;
		}

		// get the cluster type
		KernelType type;
		if (begin->name == "memory") {
			type = KernelType::Memory;
		} else {
			type = KernelType::Compute;
			bool has_output = false;
			for (auto node = IR::iterator(begin); !node.is_cluster_end(begin->cluster_head_);
			     ++node) {
				if (node->op->GetOpType() == OpType::Store ||
				    node->op->GetOpType() == OpType::Scatter) {
					has_output = true;
					break;
				}
			}

			if (!has_output) continue;
		}

		// add the cluster to the program
		program->AddKernel(type, begin);
	}

	return program;
}

}  // namespace TensorFrost