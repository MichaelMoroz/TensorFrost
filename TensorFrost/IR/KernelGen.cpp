#pragma once

#include "IR/KernelGen.h"

namespace TensorFrost {

[[nodiscard]] const Tensor* Node::GetTensor() const {
	if (tensor_->node_ != this) {
		throw std::runtime_error("Fatal Error: Tensor node does not match");
	}
	return tensor_;
}

bool CompareShape(const Node* a, const Node* b) {
	ArgMap a_shape = a->GetArgumentMap(Arg::Type::Shape);
	ArgMap b_shape = b->GetArgumentMap(Arg::Type::Shape);
	int a_dim = MaxIndexCount(a_shape);
	int b_dim = MaxIndexCount(b_shape);
	
	int min_dim = min(a_dim, b_dim);

	if (min_dim == 0) {
		return true;
	}

	if (a_dim != b_dim) {
		return false;
	}

	for (int i = 0; i < a_dim; i++) {
		Node* a_node = a_shape[i]->from_->get();
		Node* b_node = b_shape[i]->from_->get();
		//if a and b are constants, then compare their values
		if (a_node->name == "const" && b_node->name == "const") {
			if (a_node->GetTensor()->data[0] != b_node->GetTensor()->data[0]) {
				return false;
			}
		}
		//otherwise, if a and b are not the same node
		//then they are not the same shape (possibly)
		else if (a_node != b_node) {
			return false;
		}
	}

	return true;
}

// returns true if the edge between given nodes is a boundary between clusters (pre-kernels)
bool IsBoundary(const Node* input, const Node* output, int arg_index,
                Arg::Type arg_type, bool is_identity) {
	if (arg_index >= 0) {
		OpType input_type = input->op->GetOpType();
		OpType output_type = output->op->GetOpType();

		bool is_from_scatter = input_type == OpType::Scatter;
		bool is_to_scatter = output_type == OpType::Scatter;
		bool is_from_store = input_type == OpType::Store;
		bool is_from_output = input->memory_type_ == MemoryType::Output;

		if (is_from_scatter || is_from_store) {
			if (is_from_scatter && is_to_scatter) {
				return false;  // multiple scatters can be merged
			}
			return true;
		}

		if (output_type == OpType::Load || output_type == OpType::Store) {
			return arg_type == Arg::Type::Memory && !is_identity;
		}

		if (output_type == OpType::Scatter) {
			return arg_type == Arg::Type::Memory;
		}

		if (arg_type == Arg::Type::Shape) // shape must be outside cluster
			return true;
	}

	if (!CompareShape(input, output)) {
		return true;
	}

	// memory should not be inside work clusters
	if ((input->name == "memory" || output->name == "memory") &&
	    (input->name != output->name)) {
		return true;
	}

	return false;
}


void IR::Clusterize() const {
	Lable* current_cluster = nullptr;
	for (auto node = begin(); !node.is_end(); ++node) {
		// remove old cluster head
		if (node->cluster_head_ != nullptr) {
			// if last cluster node, delete lable
			if (node->next_ == nullptr ||
			    node->next_->cluster_head_ != node->cluster_head_) {
				delete node->cluster_head_;
			}
			node->cluster_head_ = nullptr;
		}

		// check if node is a cluster edge
		const Tensor* tensor = node->GetTensor();

		Arguments indices = node->GetArguments(Arg::Type::Index);
		bool identity = indices.empty();

		bool is_boundary = false;
		Node* prev = node.get_prev();
		if (prev != nullptr) {
			if (prev->cluster_head_ == current_cluster &&
			    IsBoundary(prev, *node, -1, Arg::Type::None, identity)) {
				is_boundary = true;
			}
		} else {
			is_boundary = true;
		}

		// go over all inputs
		for (auto& input : tensor->node_->inputs_) {
			// check if input is the boundary of this cluster
			if (input.from_->get()->cluster_head_ == current_cluster &&
			    IsBoundary(input.from_->get(), *node, input.index_, input.type_,
			               identity)) {
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


map<Node*, Node*> IR::CopyComputation(
    const unordered_set<Node*>& targets) const {
	// do a depth first search to copy all the nodes required for the targets
	unordered_set<Node*> nodes_to_copy;
	std::function<void(Node*)> dfs = [&](Node* node) {
		if (nodes_to_copy.find(node) != nodes_to_copy.end()) {
			return;
		}
		nodes_to_copy.insert(node);
		for (auto& input : node->inputs_) {
			if (input.type_ == Arg::Type::Shape)
				continue;
			if (input.from_->get()->name == "memory")
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
		for (Arg& arg : node->inputs_) {
			// if shape or memory argument, then no need to use copied node
			if (arg.from_->get()->name == "memory" ||
			    arg.type_ == Arg::Type::Shape) {
				new_args.push_back(arg);
				continue;
			}

			Node* from = arg.from_->get();

			if (!copied_node_map.contains(from)) {
				throw std::runtime_error("Node not found");
			}

			// create new argument
			new_args.emplace_back(arg.type_, copied_node_map[from]->GetLable(),
			                      arg.index_);
		}

		// create new node
		Tensor* tensor = Tensor::GetCopy(*node->GetTensor(), new_args);
		Node* new_node = tensor->node_;
		copied_node_map[node.get()] = new_node;
	}

	return copied_node_map;
}

void IR::OptimizeClusters() {
	// get cluster data
	ClusterProp clusters = GetClusterProperties();

	// go over each cluster and copy computations outside the cluster if they are
	// cheap enough
	for (auto* cluster_begin : clusters.cluster_heads) {
		Node* begin = cluster_begin->node_;

		if (begin->name == "memory") {
			continue;
		}

		unordered_set<Arg*> args_to_copy;
		// go over all nodes in the cluster and check if their inputs can be copied
		for (auto node = Iterator(begin); !node.is_cluster_end(cluster_begin);
		     ++node) {
			// go over all inputs
			for (auto& input : node->inputs_) {
				// if input is memory or shape, then skip
				if (input.type_ == Arg::Type::Memory ||
				    input.type_ == Arg::Type::Shape ||
				    input.from_->get()->name == "memory") {
					continue;
				}

				// if input is outside the cluster, then it can be copied
				if (input.from_->get()->cluster_head_ != node->cluster_head_) {
					// check if input is cheap enough to copy
					if (clusters.node_cost[input.from_->get()] < 256.0F) {
						args_to_copy.insert(&input);
					}
				}
			}
		}

		unordered_set<Node*> nodes_to_copy;
		for (auto* arg : args_to_copy) {
			nodes_to_copy.insert(arg->from_->get());
		}

		// copy all the nodes at the beginning of the cluster
		map<Node*, Node*> copied_node_map;
		ExecuteExpressionBefore(
		    begin, [&]() { copied_node_map = CopyComputation(nodes_to_copy); });

		// replace all the arguments that use the copied nodes
		for (auto* arg : args_to_copy) {
			arg->from_ = copied_node_map[arg->from_->get()]->GetLable();
		}
	}
}

void IR::RemoveUnusedNodes() {
	// use depth first search to find all nodes that are used for the output nodes
	unordered_set<Node*> used_nodes;

	UpdateNodeOutputs();

	//mark all output nodes as used
	for (auto node = begin(); !node.is_end(); ++node) {
		if (node->memory_type_ == MemoryType::Output) {
			used_nodes.insert(node.get());
		}
	}

	// TODO (Moroz): this might not work, beware

	// forwards pass (only for memory operations)
	for (auto node = begin(); !node.is_end(); ++node) {
		if (used_nodes.contains(node.get())) {
			continue;
		}
		// if any of the memory inputs of this node is used, then this node is used
		for (auto& input : node->inputs_) {
			if (input.type_ == Arg::Type::Memory &&
			    used_nodes.contains(input.from_->get())) {
				used_nodes.insert(node.get());
				break;
			}
		}
	}

	//backwards pass to find all nodes that are used by the output nodes
	for (auto node = end(); !node.is_begin(); --node) {
		//if any of the outputs of this node is used, then this node is used
		for (auto& output : node->outputs_) {
			if (used_nodes.contains(output->to_->get())) {
				used_nodes.insert(node.get());
				break;
			}
		}
		// if any of the memory inputs of this node is used, then this node is used
		for (auto& input : node->inputs_) {
			if (input.type_ == Arg::Type::Memory &&
				used_nodes.contains(input.from_->get())) {
				used_nodes.insert(node.get());
				break;
			}
		}
	}

	//forwards pass (only for memory operations)
	for (auto node = begin(); !node.is_end(); ++node) {
		if (used_nodes.contains(node.get())) {
			continue;
		}
		//if any of the memory inputs of this node is used, then this node is used
		for (auto& input : node->inputs_) {
			if (input.type_ == Arg::Type::Memory &&
				used_nodes.contains(input.from_->get())) {
				used_nodes.insert(node.get());
				break;
			}
		}
	}

	// remove all nodes that are not used
	unordered_set<Node*> nodes_to_remove;
	for (auto node = begin(); !node.is_end(); ++node) {
		if (node->name == "memory") {
			continue;
		}
		if (!used_nodes.contains(node.get())) {
			nodes_to_remove.insert(node.get());
		}
	}

	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}
}

ClusterProp IR::GetClusterProperties() const {
	map<Lable*, vector<Node*>> cluster_outputs;
	map<Node*, vector<Arg*>> node_output;
	map<Node*, float> node_cost;
	vector<Lable*> cluster_heads;
	unordered_set<Lable*> added_clusters;

	UpdateNodeOutputs();
	// find all nodes that are outputs of a cluster (i.e. point to a node outside
	// the cluster)
	for (auto node = begin(); !node.is_end(); ++node) {
		if (node->name == "memory" || node->cluster_head_ == nullptr) {
			continue;
		}

		float input_cost = node->op->GetCost();
		for (auto& input : node->inputs_) {
			if (input.from_->get()->cluster_head_ == node->cluster_head_ &&
			    input.type_ != Arg::Type::Memory &&
			    input.type_ != Arg::Type::Shape) {
				input_cost += node_cost[input.from_->get()];
			}
		}
		node_cost[*node] = input_cost;

		bool is_output = node->memory_type_ == MemoryType::Output;

		vector<Arg*> outputs;
		for (auto& output : node->outputs_) {
			if (output->to_ == nullptr) continue;
			// if is a shape or memory argument, then skip (shape is loaded on CPU)
			if (output->type_ == Arg::Type::Shape) {
				continue;
			}
			Node* output_node = output->to_->get();
			if (output_node->cluster_head_ != node->cluster_head_) {
				outputs.push_back(output);
				is_output = true;
			}
		}

		if (is_output) {
			cluster_outputs[node->cluster_head_].push_back(*node);
			node_output[*node] = outputs;
		}

		if (!added_clusters.contains(node->cluster_head_))
			cluster_heads.push_back(node->cluster_head_);
		added_clusters.insert(node->cluster_head_);
	}

	return ClusterProp(cluster_outputs, node_output, node_cost, cluster_heads);
}

void IR::PostProcessClusters() {
	// get cluster data
	ClusterProp clusters = GetClusterProperties();

	// go over all outputs of each cluster and create memory nodes to store the
	// output
	for (const auto& cluster_out : clusters.output) {
		vector<Node*> cluster_outs = cluster_out.second;
		Lable* cluster_head = cluster_out.first;

		for (auto* output : cluster_outs) {
			Node* mem;
			// add memory node before this cluster
			ExecuteExpressionBefore(
			    cluster_head->node_,
			    [&]() {
				    mem = Tensor::Memory(output->GetArguments(Arg::Type::Shape),
				                         output->tensor_->type)
				              .node_;

				    if (output->memory_type_ == MemoryType::Output) {
					    mem->memory_type_ = MemoryType::Output;
					    mem->memory_index_ = output->memory_index_;
					    output->memory_type_ = MemoryType::None;
				    }
			    },
			    false);

			// go over all outputs of this node and replace their input with the
			// memory node
			for (auto& arg_out : clusters.node_output[output]) {
				if (arg_out->type_ != Arg::Type::Shape &&
				    arg_out->type_ != Arg::Type::Memory) {
					// if not a memory or shape argument, then the memory needs to be
					// loaded before the node
					ExecuteExpressionBefore(arg_out->to_->get(), [&]() {
						Tensor& loaded = Tensor::Load(*mem->GetTensor());
						// loaded.node->cluster_id_ = arg_out->to_->get()->cluster_id_;
						//  the node must now use the loaded value
						arg_out->from_ = loaded.node_->GetLable();
					});
				} else {
					// otherwise the memory can be used directly
					arg_out->from_ = mem->GetLable();
				}
			}

			// add store node after this node
			ExecuteExpressionAfter(output, [&]() {
				// add store node after this node
				Tensor* store = &Tensor::Store(*mem->GetTensor(), *output->GetTensor());
				store->node_->cluster_head_ = cluster_head;
			});
		}
	}

	// replace all inputs pointing to memory nodes with the memory node
	for (auto node = begin(); !node.is_end(); ++node) {
		if (node->name == "memory") continue;

		for (auto& input : node->inputs_) {
			if (input.type_ == Arg::Type::Memory ||
			    input.type_ == Arg::Type::Shape)
				continue;

			if (input.from_->get()->name == "memory") {
				// load the memory node before this node
				ExecuteExpressionBefore(node.get(), [&]() {
					Tensor& loaded = Tensor::Load(*input.from_->get()->GetTensor());
					input.from_ = loaded.node_->GetLable();
				});
			}
		}
	}
}

vector<Tensor*> ComputeIndicesFromLinearIndex(Tensor* index, Tensors kernel_shape, int dims)
{
	vector<Tensor*> indices = vector<Tensor*>(dims);
	Tensors sizes = Tensors(dims);
	sizes[0] = kernel_shape[dims - 1];
	for (size_t i = 1; i < dims - 1; i++) {
		sizes[i] = &(*sizes[i - 1] * *kernel_shape[dims - i - 1]);
	}

	Tensor* temp;
	for (size_t i = 0; i < dims; i++) {
		Tensor* idx0 = index;
		if (i < dims - 1) {
			idx0 = &(*idx0 / *sizes[dims - i - 2]);
		}
		if (i > 0) {
			temp = &(*temp * *kernel_shape[i]);
			idx0 = &(*idx0 - *temp);
			if (i != dims - 1) temp = &(*temp + *idx0);
		} else {
			temp = idx0;
		}
		indices[i] = idx0;
	}

	return indices;
}


// compute the flat index (in C-order)
Tensor* ComputeFlatIndex(ArgMap memory_shape, vector<Tensor*> indices, map<int, const Tensor*> idx, int memory_dim) 
{
	std::function<const Tensor*(int)> get_shape = [&](int dim) {
		return memory_shape[dim]->from_->get()->GetTensor();
	};

	// function to get index for given dimension, if not found then return
	// default dim index
	std::function<Tensor*(int)> get_index = [&](int dim) {
		Tensor* out;
		if (idx.find(dim) != idx.end()) {
			out = const_cast<Tensor*>(idx[dim]);
		} else {
			out = indices[dim];
		}
		// return out; //unsafe
		return &Tensor::clamp(*out, TensorFrost::Tensor::Constant(0),
		                      *get_shape(dim) - TensorFrost::Tensor::Constant(1));
	};

	// compute the flat index (C-order)
	Tensor* flat_index = get_index(0);
	for (int i = 1; i < memory_dim; i++) {
		flat_index = &(*flat_index * *get_shape(i));
		flat_index = &(*flat_index + *get_index(i));
	}

	return flat_index;
}

void IR::LinearModeIndices(Tensor*& thread_index, vector<Tensor*>& indices, Lable* cluster_head, int dims, Tensors kernel_shape)
{
	ExecuteExpressionBefore(cluster_head->node_, [&]() {
		thread_index = &cluster_head->node_->GetTensor()->ThreadIndex();
		indices = ComputeIndicesFromLinearIndex(thread_index, kernel_shape, dims);
	});

	// replace all dim nodes with the corresponding index node
	unordered_set<Node*> nodes_to_remove;
	for (auto node = Iterator(cluster_head->node_);
	     !node.is_cluster_end(cluster_head);
			++node) {
		if (node->name == "dim_id") {
			int dim = node->GetTensor()->data[0];
			if (dim >= dims) {
				throw runtime_error("Invalid dimension index " + to_string(dim) +
									" for kernel of size " + to_string(dims));
			}

			// swap the dim node with the corresponding index node
			CopyLable(node.get(), indices[dim]->node_);

			// remove the dim node
			nodes_to_remove.insert(node.get());
		}
	}

	// remove all dim nodes
	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}
}

void IR::MultiDimensionalModeIndices(Tensor*& thread_index, vector<Tensor*>& indices, Lable* cluster_head, int dims, Tensors kernel_shape)
{
	// just use dim_id nodes as indices
	for (auto node = Iterator(cluster_head->node_);
		!node.is_cluster_end(cluster_head);
		++node) {
		if (node->name == "dim_id") {
			int dim = node->GetTensor()->data[0];
			if (dim >= dims) {
				throw runtime_error("Invalid dimension index " + to_string(dim) +
													" for kernel of size " + to_string(dims));
			}

			indices[dim] = const_cast<Tensor*>(node->GetTensor());
		}
	}

	//add dim_id nodes if they are missing
	for (int i = 0; i < dims; i++) {
		if (indices[i] == nullptr) {
			ExecuteExpressionBefore(cluster_head->node_, [&]() {
				indices[i] = &cluster_head->node_->GetTensor()->Index(i);
			});
		}
	}
}

void IR::TransformToLinearIndex() {
	ClusterProp clusters = GetClusterProperties();

	// replace all dim_id nodes with the corresponding index computed from
	// thread_id
	for (auto* cluster_begin : clusters.cluster_heads) {
		// load kernel shape
		map<int, const Tensor*> kernel_shape_map =
		    cluster_begin->node_->GetArgumentTensors(Arg::Type::Shape);
		Tensors kernel_shape;
		for (auto& shape : kernel_shape_map) {
			kernel_shape.push_back(shape.second);
		}

		if (kernel_shape.empty()) {
			// can skip if no kernel shape - no index
			continue;
		}

		//TODO (Moroz): make cluster lables separate from node lables

		// compute the index for each dimension
		size_t dims = kernel_shape.size();
		Tensor* thread_index;
		vector<Tensor*> indices = vector<Tensor*>(dims);

		switch (indexing_mode_)
		{ 
		case KernelIndexingMode::Linear:
			LinearModeIndices(thread_index, indices, cluster_begin, dims, kernel_shape);
			break;
		case KernelIndexingMode::MultiDimensional:
			MultiDimensionalModeIndices(thread_index, indices, cluster_begin, dims, kernel_shape);
			break;
		default:
			throw runtime_error("Invalid indexing mode");
		}

		// go over all nodes that take an index as input (e.g. load, store, atomic)
		for (auto node = Iterator(cluster_begin->node_);
		     !node.is_cluster_end(cluster_begin);
		     ++node) {
			auto op_type = node->op->GetOpType();
			if (op_type == OpType::Load || op_type == OpType::Store ||
			    op_type == OpType::Scatter) {
				ExecuteExpressionBefore(*node, [&]() {
					// get the input memory node
					const Tensor* memory =
					    node.get()->GetArgumentTensors(Arg::Type::Memory)[0];

					ArgMap memory_shape = memory->node_->GetArgumentMap(Arg::Type::Shape);
					int memory_dim = MaxIndexCount(memory_shape);

					// get the index nodes
					map<int, const Tensor*> idx =
					    node->GetArgumentTensors(Arg::Type::Index);

					//just use the thread index if no index is provided
					if (idx.empty() && indexing_mode_ == KernelIndexingMode::Linear) {
						// add the thread index node edge
						node->AddArgument(thread_index->node_, Arg::Type::Index, 0);
						return;
					}

					Tensor* flat_index = ComputeFlatIndex(memory_shape, indices, idx, memory_dim);

					// TODO(Moroz): add different modes for clamping (e.g. clamp, wrap,
					// mirror, zero)

					// remove the index node edges
					node->RemoveArguments(Arg::Type::Index);

					// add the flat index node edge
					node->AddArgument(flat_index->node_, Arg::Type::Index, 0);
				});
			}
		}
	}
}

Program* GenerateProgram(IR* ir) {
	auto* program = new Program(ir);

	// go over all clusters, find their type, and add them to the program if they
	// are used
	for (auto node = ir->begin(); !node.is_end(); ++node) {
		Node* begin = node.get();

		if (begin->name != "memory" && !node.is_cluster_begin()) {
			continue;
		}

		// get the cluster type
		KernelType type;
		map<Node*, int> variables;
		map<Node*, int> memory_nodes;
		ArgMap shape;
		int dim = 0;
		if (begin->name == "memory") {
			if (begin->memory_type_ == MemoryType::Input) {
				continue;
			}
			type = KernelType::Memory;
		} else {
			type = KernelType::Compute;
		
			bool has_output = false;
			bool has_shape = false;

			int variable_index = 0;
			int memory_index = 0;

			for (auto node = IR::Iterator(begin);
			     !node.is_cluster_end(begin->cluster_head_); ++node) {
				OpType op_type = node->op->GetOpType();
				if (op_type == OpType::Store || op_type == OpType::Scatter) {
					has_output = true;
				}
				if (op_type == OpType::Load || op_type == OpType::Store ||
				    op_type == OpType::Scatter) {
					// get the memory node
					const Tensor* memory =
					    node->GetArgumentTensors(Arg::Type::Memory)[0];

					if (!memory_nodes.contains(memory->node_))
					{
						memory_nodes[memory->node_] = memory_index++;
					}
				}

				// get all input arguments
				map<int, const Tensor*> inputs =
				    node->GetArgumentTensors(Arg::Type::Input);
				for (auto& input : inputs) {
					if (input.second->node_->name == "memory") {
						if (!variables.contains(input.second->node_))
						{
							variables[input.second->node_] = variable_index++;
						}
					}
				}

				if (!has_shape)
				{
					// get the shape argument (if exists)
					shape = node->GetArgumentMap(Arg::Type::Shape);
					dim = MaxIndexCount(shape);
					if (dim != 0 && shape.size() == dim) {
						has_shape = true;
					}
				}
			}

			if (!has_output) continue;

			if (!has_shape)
			{
				throw runtime_error("Kernel does not have a shape");
			}
		}

		// add the cluster to the program
		program->AddKernel(type, ir->indexing_mode_, begin, variables, memory_nodes,
		                   shape, dim);
	}

	return program;
}

}  // namespace TensorFrost