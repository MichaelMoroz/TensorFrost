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
		// if a and b are constants, then compare their values
		if (a_node->name == "const" && b_node->name == "const") {
			if (a_node->GetTensor()->data[0] != b_node->GetTensor()->data[0]) {
				return false;
			}
		}
		// otherwise, if a and b are not the same node
		// then they are not the same shape (possibly)
		else if (a_node != b_node) {
			return false;
		}
	}

	return true;
}

// returns true if the edge between given nodes is a boundary between kernels
bool IsBoundary(const Node* input, const Node* output, bool is_identity = true, int arg_index = -1, Arg::Type arg_type = Arg::Type::None) {
	if (arg_index >= 0) {
		const Operation* input_op = input->op;
		const Operation* output_op = output->op;

		if (output_op->HasAnyType(OpType::Load, OpType::Store)) {
			return arg_type == Arg::Type::Memory && !is_identity;
		}

		if (output_op->HasAnyType(OpType::Scatter)) {
			return arg_type == Arg::Type::Memory;
		}

		if (arg_type == Arg::Type::Shape) // shape must be outside kernels
			return true;

		//if input has changed the memory and the output is a load then it is a boundary
		if (input_op->HasAllTypes(OpType::MemoryOp, OpType::Modifier) &&
		    output_op->HasAnyType(OpType::Load)) {
			return true;
		}
	}

	if (!CompareShape(input, output)) {
		return true;
	}

	// memory should not be inside work kernels
	if ((input->name == "memory" || output->name == "memory") &&
	    (input->name != output->name)) {
		return true;
	}

	return false;
}


void IR::SeparateOperationsIntoKernels() const {
	vector<Cluster*> clusters;
	Cluster* current_cluster = nullptr;

	RecomputeGlobalIndices();
	UpdateNodeOutputs();

	for (auto node = begin(); !node.is_end(); ++node) {
		// remove old cluster head
		if (node->cluster_ != nullptr) {
			// if last cluster node, delete lable
			if (node->next_ == nullptr ||
			    node->next_->cluster_ != node->cluster_) {
				delete node->cluster_;
			}
			node->cluster_ = nullptr;
		}

		// check if node is a cluster edge
		const Tensor* tensor = node->GetTensor();

		Arguments indices = node->GetArguments(Arg::Type::Index);
		//TODO: do a pass before - removing MemoryOp's by local ops if they have no indices
		bool identity = indices.empty();

		bool is_boundary = false;
		Node* prev = node.get_prev();
		if (prev != nullptr) {
			if (prev->cluster_ == current_cluster &&
			    IsBoundary(prev, *node, identity)) {
				is_boundary = true;
			}
		} else {
			is_boundary = true;
		}

		//TODO (Moroz): do separately on all nodes after clusterization
		if (current_cluster != nullptr && current_cluster->shape_node_ != nullptr) {
			if (!CompareShape(current_cluster->shape_node_->get(), node.get())) {
				is_boundary = true;
			}
		}

		// go over all inputs
		for (auto& input : tensor->node_->inputs_) {
			// get latest input version
			const Node* latest = input.from_->get()->GetLastVersion(*node);
			// check if input is the boundary of this cluster
			if (latest->cluster_ == current_cluster &&
			    IsBoundary(latest, *node, identity, input.index_, input.type_)) {
				is_boundary = true;
				break;
			}
		}

		if (is_boundary) {
			current_cluster = new Cluster(*node);
			clusters.push_back(current_cluster);
		}

		node->cluster_ = current_cluster;

		if (current_cluster->shape_node_ == nullptr) {
			// TODO (Moroz): do operation categories
			if (!node->op->HasAnyType(OpType::Function, OpType::Operator, OpType::Load, OpType::Store, OpType::Scatter))
				continue;

			// get the shape argument (if exists)
			ArgMap shape = node->GetArgumentMap(Arg::Type::Shape);
			int dim = MaxIndexCount(shape);
			if (dim != 0 && shape.size() == dim) {
				current_cluster->shape_node_ = node->GetLable();
			}
		}
	}

	// update cluster shape node if absent
	for (auto* cluster : clusters) {
		if (cluster->shape_node_ == nullptr) {
			cluster->shape_node_ = cluster->begin_->GetLable();
		}
	}
}

void IR::PrintListing(string name, bool compact,
                      map<Node*, string> invalid_nodes) const {
#ifdef NDEBUG
	return;
#endif
	string listing = GetOperationListing(*this, false, invalid_nodes) + "\n\n";

	if (!invalid_nodes.empty()) {
		listing += "Step [" + name + "] failed. ";
		throw std::runtime_error(listing);
	} else {
		cout << "Step [" << name << "] completed successfully: \n" << endl;
		cout << listing << endl;
	}
}

bool BoundaryValid(const Node* input, const Node* output,
                   bool is_identity = true, int arg_index = -1,
                   Arg::Type arg_type = Arg::Type::None) {
	bool same_cluster = input->cluster_ == output->cluster_;
	bool is_boundary = IsBoundary(input, output, is_identity, arg_index, arg_type);
	if (!same_cluster) return true;
	return !is_boundary;
}

void IR::RecomputeGlobalIndices() const {
	// go over all nodes and recompute global indices
	int index = 0;
	for (auto node = begin(); !node.is_end(); ++node) {
		node->global_index_ = index++;
	}
}

void IR::CheckIR(string name, bool check_clustering, bool check_kernels) const {
#ifdef NDEBUG
	return;
#endif
	RecomputeGlobalIndices();

	map<Node*, string> invalid_nodes;
	//check if the IR is clusterized correctly
	for (auto node = begin(); !node.is_end(); ++node) {
		// check if node is a cluster edge
		const Tensor* tensor = node->GetTensor();

		Arguments indices = node->GetArguments(Arg::Type::Index);
		bool identity = indices.empty();

		Node* prev = node.get_prev();

		if (prev == nullptr) continue;

		if (check_clustering) {
			if (!BoundaryValid(prev, *node, identity)) {
				invalid_nodes[node.get()] = "Invalid node order";
			}
		}

		// go over all inputs
		for (auto& input : tensor->node_->inputs_) {
			Node* from = input.from_->get();
			Node* to = node.get();

			if (check_clustering)
			{
				// get latest input version
				const Node* latest = input.from_->get()->GetLastVersion(*node);
				// check if input is the boundary of this cluster
				if (!BoundaryValid(latest, to, identity, input.index_, input.type_)) {
					invalid_nodes[to] = "Invalid clusterization for argument " + Arg::TypeToString(input.type_) + ":" + to_string(input.index_);
				}
			}

			if (check_kernels)
			{
				//check if no inputs are outside the cluster
				if (from->cluster_ != to->cluster_ && 
					input.type_ != Arg::Type::Memory && 
					input.type_ != Arg::Type::Shape && from->name != "memory" &&
				    from->name != "const") {
					invalid_nodes[to] = "Argument " + Arg::TypeToString(input.type_) + ":" + to_string(input.index_) + " is outside the kernel";
				}
			}

			// check if inputs are before the node
			if (from->global_index_ >= to->global_index_) {
				invalid_nodes[to] = "Argument " + Arg::TypeToString(input.type_) + ":" + to_string(input.index_) + " is after the node";
			}
		}
	}

	PrintListing(name, false, invalid_nodes);
}

bool CannotCopyArgument(Arg& arg) {
	return arg.type_ == Arg::Type::Memory || arg.type_ == Arg::Type::Shape ||
	       arg.from_->get()->name == "memory" ||
	       arg.from_->get()->HasBeenModified();
			//(arg.from_->get()->cluster_ != arg.to_->get()->cluster_ && arg.from_->get()->name != "const"); //only copy inside the same cluster
}

map<Node*, Node*> IR::CopyComputation(
    const unordered_set<Node*>& targets) const {
	// do a depth first search to copy all the nodes required for the targets (only if in the same cluster)
	unordered_set<Node*> nodes_to_copy;
	std::function<void(Node*)> dfs = [&](Node* node) {
		if (nodes_to_copy.find(node) != nodes_to_copy.end()) {
			return;
		}
		nodes_to_copy.insert(node);
		for (auto& input : node->inputs_) {
			if (CannotCopyArgument(input)) continue;
			dfs(input.from_->get());
		}
	};

	for (Node* target : targets) {
		dfs(target);
	}

	if (nodes_to_copy.empty()) {
		return {};
	}

	if (nodes_to_copy.size() > 100)
	{
		throw std::runtime_error("Copying too many nodes, something is probably wrong");
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
			if (CannotCopyArgument(arg)) {
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

void IR::OptimizeKernels() {
	// get cluster data
	ClusterProp clusters = GetClusterProperties();

	// go over each cluster and copy computations outside the cluster if they are
	// cheap enough
	for (auto* cluster_begin : clusters.clusters) {
		Node* begin = cluster_begin->begin_;

		if (begin->name == "memory") {
			continue;
		}

		unordered_set<Arg*> args_to_copy;
		// go over all nodes in the cluster and check if their inputs can be copied
		for (auto node = Iterator(begin); !node.is_cluster_end(cluster_begin);
		     ++node) {
			// go over all inputs
			for (auto& input : node->inputs_) {
				if (CannotCopyArgument(input)) continue;

				// if input is outside the cluster and has the same shape as the node,
				// then copy it
				if (input.from_->get()->cluster_ != node->cluster_/* && CompareShape(input.from_->get(), node.get())*/) {
					// check if input is cheap enough to copy
					int input_cost = input.from_->get()->cost_;
					if (input_cost == -1.0)
					{
						throw std::runtime_error("Cost has not been computed");
					}
					if (input_cost >= 0.0F && input_cost < 256.0F) {
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

bool isConstantAndEqualTo(const Tensor* tensor, float value) {
	if (tensor->node_->name != "const" || tensor->node_->has_been_modified_) {
		return false;
	}

	switch (tensor->type) {
		case DataType::Float:
			return AsFloat(tensor->data[0]) == value;
		case DataType::Int:
			return AsInt(tensor->data[0]) == value;
		case DataType::Uint:
			return tensor->data[0] == value;
		default:
			throw std::runtime_error("Unexpected type in isConstantAndEqualTo");
	}
}

bool isConstant(const Tensor* tensor) {
	return tensor->node_->name == "const" && !tensor->node_->has_been_modified_;
}

Tensor* ApplyMultiOP(const Tensor* a, const Tensor* b, std::function<float(float, float)> opF32, std::function<int(int, int)> opI32, std::function<uint(uint, uint)> opU32) {
	switch (a->type) {
		case DataType::Float:
			return &Tensor::Constant(opF32(AsFloat(a->data[0]), AsFloat(b->data[0])));
		case DataType::Int:
			return &Tensor::Constant(opI32(AsInt(a->data[0]), AsInt(b->data[0])));
		case DataType::Uint:
			return &Tensor::Constant(opU32(a->data[0], b->data[0]));
		default:
			throw std::runtime_error("Unexpected type in ApplyMultiOP");
	}
}

#define ApplyOP(v1, v2, op) ApplyMultiOP(v1, v2, [](float a, float b) { return a op b; }, [](int a, int b) { return a op b; }, [](uint a, uint b) { return a op b; })
#define ApplyFUNC(v1, v2, func) ApplyMultiOP(v1, v2, [](float a, float b) { return func(a, b); }, [](int a, int b) { return func(a, b); }, [](uint a, uint b) { return func(a, b); })

void IR::OptimizeOperations() 
{
	for (auto node = begin(); !node.is_end(); ++node) {
		if (node->op->op_types_[0] != OpType::Operator) {
			continue;
		}

		//get node operation
		const string op = node->name;

		//get inputs
		map<int, const Tensor*> inputs = node->GetArgumentTensors(Arg::Type::Input);
		ExecuteExpressionAfter(*node, [&]() {
			const Tensor* result = nullptr;
			if (op == "add") {
				// if any are zero, replace with the other
				if (isConstantAndEqualTo(inputs[0], 0.0F)) {
					// replace with input 1
					result = inputs[1];
				} else if (isConstantAndEqualTo(inputs[1], 0.0F)) {
					// replace with input 0
					result = inputs[0];
				}

				// if all are constants, replace with result
				if (isConstant(inputs[0]) && isConstant(inputs[1])) {
					// replace with result
					result = ApplyOP(inputs[0], inputs[1], +);
				}
			} else if (op == "sub") {
				// if any are zero, replace with the other
				if (isConstantAndEqualTo(inputs[0], 0.0F)) {
					// replace with negation of input 1
					result = &(-*inputs[1]);
				} else if (isConstantAndEqualTo(inputs[1], 0.0F)) {
					// replace with input 0
					result = inputs[0];
				}

				// if all are constants, replace with result
				if (isConstant(inputs[0]) && isConstant(inputs[1])) {
					// compute result
					result = ApplyOP(inputs[0], inputs[1], -);
				}
			} else if (op == "mul") {
				// if any are zero, replace with zero
				if (isConstantAndEqualTo(inputs[0], 0.0F) ||
									    isConstantAndEqualTo(inputs[1], 0.0F)) {
					// replace with zero
					result = &Tensor::Constant(0u, inputs[0]->type);
				}

				// if any are one, replace with the other
				if (isConstantAndEqualTo(inputs[0], 1.0F)) {
					// replace with input 1
					result = inputs[1];
				} else if (isConstantAndEqualTo(inputs[1], 1.0F)) {
					// replace with input 0
					result = inputs[0];
				}

				// if all are constants, replace with result
				if (isConstant(inputs[0]) && isConstant(inputs[1])) {
					// compute result
					result = ApplyOP(inputs[0], inputs[1], *);
				}
			} else if (op == "div") {
				// if first is zero, replace with zero
				if (isConstantAndEqualTo(inputs[0], 0.0F)) {
					// replace with zero
					result = &Tensor::Constant(0u, inputs[0]->type);
				}

				// if second is one, replace with first
				if (isConstantAndEqualTo(inputs[1], 1.0F)) {
					// replace with input 0
					result = inputs[0];
				}

				// if all are constants, replace with result	
				if (isConstant(inputs[0]) && isConstant(inputs[1])) {
					// compute result
					result = ApplyOP(inputs[0], inputs[1], /);
				}
			}

			//TODO (Moroz): add more optimizations

			// if computed optimized result, replace all node references with it
			if (result != nullptr)
			{
				CopyLable(node.get(), result->node_);
			}
		});
	}
}

bool IsChangingInput(Arg* arg) {
	return (arg->type_ == Arg::Type::Memory && arg->to_->get()->op->HasAllTypes(OpType::Modifier)) ||
		   (arg->to_->get()->name == "loop_end"); //okay, this is stupid
}

void IR::RemoveUnusedOperations() {
	// use depth first search to find all nodes that are used for the output nodes
	unordered_set<Node*> used_nodes;

	UpdateNodeOutputs();

	std::function<void(Node*)> dfs = [&](Node* node) 
	{
		if (used_nodes.contains(node)) {
			return;
		}

		used_nodes.insert(node);

		//all inputs of this node are used
		for (auto& argument : node->inputs_) {
			dfs(argument.from_->get());
		}

		//if the node is a memory node or used as memory, then all outputs are used
		for (auto& argument : node->outputs_) {
			if (IsChangingInput(argument)) {
				dfs(argument->to_->get());
			}
		}
	};

	//mark all output nodes as used
	for (auto node = begin(); !node.is_end(); ++node) {
		if (node->memory_type_ == MemoryType::Output) {
			dfs(node.get());
		}
	}


	// remove all nodes that are not used
	unordered_set<Node*> nodes_to_remove;
	for (auto node = begin(); !node.is_end(); ++node) {
		if (!used_nodes.contains(node.get())) {
			nodes_to_remove.insert(node.get());
		}
	}

	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}
}

ClusterProp IR::GetClusterProperties() const {
	map<Cluster*, vector<Node*>> cluster_outputs;
	map<Node*, vector<Arg*>> node_output;
	vector<Cluster*> clusters;
	unordered_set<Cluster*> added_clusters;

	UpdateNodeOutputs();
	// find all nodes that are outputs of a cluster (i.e. point to a node outside
	// the cluster)
	for (auto node = begin(); !node.is_end(); ++node) {
		if (node->name == "memory" || node->cluster_ == nullptr || *node == nullptr) {
			continue;
		}

		float input_cost = node->op->GetCost();
		for (auto& input : node->inputs_) {
			if (input.type_ != Arg::Type::Memory &&
			    input.type_ != Arg::Type::Shape) {
				input_cost += abs(input.from_->get()->cost_);
			}
		}
		node->cost_ = input_cost;

		bool is_output = node->memory_type_ == MemoryType::Output;

		vector<Arg*> outputs;
		for (auto& output : node->outputs_) {
			if (output->to_ == nullptr) continue;
			// if is a shape or memory argument, then skip (shape is loaded on CPU)
			if (output->type_ == Arg::Type::Shape) {
				continue;
			}
			Node* output_node = output->to_->get();
			if (output_node->cluster_ != node->cluster_) {
				outputs.push_back(output);
				is_output = true;
			}
		}

		if (is_output) {
			cluster_outputs[node->cluster_].push_back(*node);
			node_output[*node] = outputs;
		}

		if (!added_clusters.contains(node->cluster_))
			clusters.push_back(node->cluster_);
		added_clusters.insert(node->cluster_);
	}

	return ClusterProp(cluster_outputs, node_output, clusters);
}

void IR::AddKernelGlobalMemoryOperations() {
	// get cluster data
	ClusterProp clusters = GetClusterProperties();

	// go over all outputs of each cluster and create memory nodes to store the
	// output
	for (const auto& cluster_out : clusters.output) {
		vector<Node*> cluster_outs = cluster_out.second;
		Cluster* cluster_ = cluster_out.first;

		for (auto* output : cluster_outs) {
			Node* mem;
			// add memory node before this cluster
			ExecuteExpressionBefore(
			    cluster_->begin_,
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
				store->node_->cluster_ = cluster_;
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
Tensor* ComputeFlatIndex(ArgMap memory_shape, vector<Tensor*> indices, map<int, const Tensor*> idx, int memory_dim, TensorIndexingMode mode = TensorIndexingMode::Clamp)
{
	if (memory_dim == 0)
	{
		return &Tensor::Constant(0);
	}

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

		switch (mode)
		{
			case TensorIndexingMode::Clamp:
				return &Tensor::clamp(
				    *out, TensorFrost::Tensor::Constant(0),
				    *get_shape(dim) - TensorFrost::Tensor::Constant(1));
			case TensorIndexingMode::Unsafe:
				return out;
			default: //TODO (Moroz): add other modes
				throw std::runtime_error("Invalid tensor indexing mode");
		}
	};

	// compute the flat index (C-order)
	Tensor* flat_index = get_index(0);
	for (int i = 1; i < memory_dim; i++) {
		flat_index = &(*flat_index * *get_shape(i));
		flat_index = &(*flat_index + *get_index(i));
	}

	return flat_index;
}

void IR::LinearModeIndices(Tensor*& thread_index, vector<Tensor*>& indices, Cluster* cluster, int dims, Tensors kernel_shape)
{
	ExecuteExpressionBefore(cluster->begin_, [&]() {
		thread_index = &cluster->shape_node_->get()->GetTensor()->ThreadIndex();
		indices = ComputeIndicesFromLinearIndex(thread_index, kernel_shape, dims);
	});

	// replace all dim nodes with the corresponding index node
	unordered_set<Node*> nodes_to_remove;
	for (auto node = Iterator(cluster->begin_);
	     !node.is_cluster_end(cluster);
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

void IR::MultiDimensionalModeIndices(Tensor*& thread_index, vector<Tensor*>& indices, Cluster* cluster_, int dims, Tensors kernel_shape)
{
	//add dim_id nodes at the beginning of the cluster
	ExecuteExpressionBefore(cluster_->begin_, [&]() {
		for (int i = 0; i < dims; i++) {
			indices[i] = &cluster_->begin_->GetTensor()->Index(i);
		}
	});
}

void IR::FinalizeMemoryIndexing() {
	ClusterProp clusters = GetClusterProperties();

	for (auto* cluster : clusters.clusters) {
		Node* shape_node = cluster->shape_node_->get();
		if (shape_node == nullptr) continue;
		// load kernel shape
		map<int, const Tensor*> kernel_shape_map =
		    shape_node->GetArgumentTensors(Arg::Type::Shape);
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
			LinearModeIndices(thread_index, indices, cluster, dims, kernel_shape);
			break;
		case KernelIndexingMode::MultiDimensional:
		case KernelIndexingMode::MultiDimensionalBlocks: //TODO (Moroz): add proper support for blocks
			MultiDimensionalModeIndices(thread_index, indices, cluster, dims, kernel_shape);
			break;
		default:
			throw runtime_error("Invalid kernel indexing mode");
		}

		// go over all nodes that take an index as input (e.g. load, store, atomic)
		for (auto node = Iterator(cluster->begin_);
		     !node.is_cluster_end(cluster);
		     ++node) {
			if (node->op->HasAllTypes(OpType::MemoryOp)) {
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

					Tensor* flat_index = ComputeFlatIndex(memory_shape, indices, idx, memory_dim, tensor_indexing_mode_);

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

void IR::CompileIR() 
{
	// TODO (Moroz): Make sure that shape works with non-const tensors
	// TODO (Moroz): Add auto tests into build system

	SetKernelIndexingMode(KernelIndexingMode::MultiDimensional);
	SetTensorIndexingMode(TensorIndexingMode::Clamp);

	CheckIR("Input", false, false);
	OptimizeOperations();
	CheckIR("Optimize operations", false, false);
	RemoveUnusedOperations();
	SeparateOperationsIntoKernels();
	CheckIR("Separate Operations Into Kernels", true, false);
	OptimizeKernels();
	CheckIR("Optimize Kernels", true, false);
	RemoveUnusedOperations();
	CheckIR("Remove Unused Operations 1", true, false);
	AddKernelGlobalMemoryOperations();
	CheckIR("Add Kernel Global Memory Operations", true, true);
	FinalizeMemoryIndexing();
	OptimizeKernels();
	CheckIR("Finalize Memory Indexing", true, true);
	OptimizeOperations();
	RemoveUnusedOperations();
	CheckIR("Remove Unused Operations 2", true, true);
}

Program* GenerateProgram(IR* ir) 
{
	ir->CompileIR();

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
			     !node.is_cluster_end(begin->cluster_); ++node) {
				if (node->op->HasAllTypes(OpType::MemoryOp, OpType::Modifier)) {
					has_output = true;
				}
				if (node->op->HasAllTypes(OpType::MemoryOp)) {
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