#include "IR/KernelGen.h"

namespace TensorFrost {

[[nodiscard]] const Tensor* Node::GetTensor() const {
	if (tensor_->node_ != this) {
		throw std::runtime_error("Fatal Error: Tensor node does not match");
	}
	return tensor_;
}

void ArgumentManager::UpdateArgument(ArgID id, Node *node) {
	if(node == nullptr) {
		throw std::runtime_error("Node is null");
	}
	if(!Has(id)) {
		throw std::runtime_error("No argument to update");
	}
	inputs_[id] = node;
	argument_types_[id] = node->type;
}

//if shape nodes are compatible, then return the broadcast shape, if not return nullptr
ShapeDimCompareResult CompareShapeDim(Node* a_node, Node* b_node, bool exact_match) {
	ShapeDimCompareResult result;
	result.broadcast = false;
	result.a_dim = -1;
	result.b_dim = -1;
	if (a_node->name == "const") result.a_dim = a_node->data[0];
	if (b_node->name == "const") result.b_dim = b_node->data[0];

	// if one of the nodes is a constant = 1, then it is a broadcast
	if ((result.a_dim == 1 || result.b_dim == 1) && !(result.a_dim == 1 && result.b_dim == 1) && !exact_match) {
		result.compatible = true;
		result.broadcast = true;
		if (result.a_dim == 1) {
			result.broadcast_dim = b_node;
			return result;
		} else {
			result.broadcast_dim = a_node;
			return result;
		}
	}

	// if a and b are constants, then compare their values
	if (result.a_dim != -1 && result.b_dim != -1) {
		if (result.a_dim != result.b_dim) {
			result.compatible = false;
			return result;
		} else {
			result.compatible = true;
			result.broadcast_dim = a_node;
			return result;
		}
	}

	// otherwise, if a and b are not the same node then they are not the same
	// shape (possibly)
	if (a_node != b_node) {
		result.compatible = false;
		return result;
	}

	result.compatible = true;
	result.broadcast_dim = a_node;
	return result;
}

ShapeCompareResult CompareShape(ShapeInfo& a, ShapeInfo& b, bool exact_match, bool throw_error) {
	ShapeCompareResult result;
	result.compatible = true;
	result.broadcast = false;
	result.a_dim = a.dim;
	result.b_dim = b.dim;
	result.broadcast_dim = max(a.dim, b.dim);

	int min_dim = min(a.dim, b.dim);

	if (exact_match && min_dim > 0) {
		if (a.dim != b.dim) {
			result.compatible = false;
			if (throw_error) {
				throw std::runtime_error("Shapes must have the same dimension for " +
				                         a.name + " and " + b.name);
			}
			return result;
		}
	}

	for (int i = 0; i < min_dim; i++) {
		Node* a_node = a.shape[a.dim - i - 1];
		Node* b_node = b.shape[b.dim - i - 1];
		int broadcast_index = max(a.dim, b.dim) - i - 1;

		ShapeDimCompareResult res = CompareShapeDim(a_node, b_node, exact_match);

		if(!res.compatible) {
			result.compatible = false;
			if (throw_error) {
				if(res.a_dim != -1 || res.b_dim != -1) {
					throw std::runtime_error("Shapes are not compatible for nodes: " + a.name + " and " + b.name + " with constant values " + to_string(res.a_dim) + " and " + to_string(res.b_dim) + " at index " + to_string(i));
				}
				throw std::runtime_error("Shapes are potentially not compatible for nodes: " + a.name + " and " + b.name + " at index " + to_string(i));
			}
			return result;
		}

		if(res.broadcast) {
			result.broadcast = true;
		}

		result.broadcast_shape.AddShape(broadcast_index, res.broadcast_dim);
	}

	//add the rest of the broadcast shape
	for (int i = min_dim; i < result.broadcast_dim; i++) {
		result.broadcast = true;
		int broadcast_index = max(a.dim, b.dim) - i - 1;
		if (a.dim > b.dim) {
			result.broadcast_shape.AddShape(broadcast_index, a.shape[a.dim - i - 1]);
		} else {
			result.broadcast_shape.AddShape(broadcast_index, b.shape[b.dim - i - 1]);
		}
	}

	if (result.broadcast_shape.dim != result.broadcast_dim) {
		throw std::runtime_error("Internal Error: Broadcast shape does not match the broadcast dim");
	}

	return result;
}


ShapeCompareResult CompareShape(const Node* a, const Node* b, bool exact_match, bool throw_error) {
	ShapeInfo a_info = ShapeInfo(a);
	ShapeInfo b_info = ShapeInfo(b);
	return CompareShape(a_info, b_info, exact_match, throw_error);
}

//TODO: rewrite this function
void IR::SeparateOperationsIntoKernels() {
	auto kernel_scopes = KernelScope::ComputeScopes(root);

	// create kernel nodes for all kernel scopes
	for (auto scope : kernel_scopes.first) {
		// create kernel node before the scope
		ExecuteExpressionBefore(scope->begin, [&]() {
			//create kernel node
			Tensor& tensor = Tensor::Kernel(scope->scope_shape.GetTensors());
		 	Node* kernel_node = tensor.node_;
		 	// make the scope nodes children of the kernel node
		 	kernel_node->child = scope->begin;
			kernel_node->next = scope->end->next;
		 	scope->begin->parent = kernel_node;
		 	scope->begin->prev = nullptr;
		 	scope->end->next->prev = kernel_node;
		 	scope->end->next = nullptr;
		});
	}

	UpdateGraph();
}

string IR::PrintListing(map<Node*, string> node_debug) const {
	return GetOperationListing(*this, false, node_debug) + "\n\n";
}

// check if all child nodes in a kernel have compatible shape to the kernel
void IR::CheckKernelShapes() {
	// get kernels
	vector<Node*> kernels = GetNodesOfType("kernel");

	// go over all outputs of each kernel and create memory nodes to store the
	// output
	for (auto kernel : kernels) {
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			// check if the node has a shape argument
			ShapeCompareResult result = CompareShape(kernel, node.get(), false, true);
		}
	}

	UpdateGraph();
}

void IR::RemoveNode(Node* node) {
	if (node->valid()) {
		// if child node exists, iterate through it and remove all children
		if (node->child) {
			vector<Node*> to_delete;
			for (auto child = NodeIterator(node); !child.end(); child.next()) {
				to_delete.push_back(*child);
			}
			for (Node* child : to_delete) {
				RemoveNode(child);
			}
		}

		// if direct child of its parent
		if (node->parent && node->parent->child == node) {
			node->parent->child = node->next;
		} else if (node->prev) {
			node->prev->next = node->next;
		}

		node->next->prev = node->prev;
		delete node;
	}
}

void IR::CheckIR(string name, bool check_clustering, bool check_kernels) const {
#ifdef NDEBUG
	return;
#endif
	UpdateGraph();

	map<Node*, string> invalid_nodes;
	//check if the IR is clusterized correctly
	for (auto node = begin(); !node.end(); node.next()) {
		bool identity = node->args.Count(ArgType::Index) == 0;

		Node* prev = node->prev;

		if (prev == nullptr) continue;


		// go over all inputs
		for (auto& [id, input] : node->args.inputs_) {
			Node* to = node.get();

			// check if inputs are before the node
			if (input->index_ >= to->index_ && input->name != "const") {
				if (id.first != ArgType::Shape) {
					invalid_nodes[to] = "Argument " + TypeToString(id.first) + ":" +
										to_string(id.second) + " is after the node";
				}
			}
		}
	}

	string listing = PrintListing(invalid_nodes);

	if (!invalid_nodes.empty()) {
		listing += "Step [" + name + "] failed. ";
		throw std::runtime_error(listing);
	} else {
		cout << "Step [" << name << "] completed successfully: \n" << endl;
		cout << listing << endl;
	}
}

void IR::ReorderOperations() {
	// get kernel data
	vector<Node*> kernels = GetNodesOfType("kernel");

	for (auto* kernel: kernels) {
		unordered_set<Node*> nodes_to_move;
		// go over all nodes in the kernel and check if their inputs can be copied
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			// go over all inputs
			for (auto& [id, from] : node->args.inputs_) {
				bool outside_kernel = !from->HasParent(kernel);
				if (outside_kernel && !node->args.CannotMoveArgument(id)) {
					// if this node is a set and its input is outside of the cluser ->
					// move it inside
					if (node->op->HasAllTypes(OpClass::Set)) {
						nodes_to_move.insert(from);
					}
				}
			}
		}

		//TODO (Moroz): do a check on order of the moved nodes - seems to be breaking sometimes

		// move all the nodes that are outside the kernel inside
		Node* kernel_begin = kernel->child;
		for (auto* node : nodes_to_move) {
			MoveNodeTo(kernel_begin, node);
		}
	}

	UpdateGraph();
}

/// <summary>
/// Copy given nodes
/// </summary>
/// <param name="nodes_to_copy">target nodes to copy</param>
/// <param name="argument_replacements">if given, the arguments to replace</param>
/// <param name="indices">if given, the indices to use</param>
/// <param name="targets">if given, the target nodes</param>
/// <param name="must_copy_all">if true, all nodes and their arguments must be copied</param>
/// <returns>mappings between the original nodes and the copied nodes</returns>
map<Node*, Node*> IR::CopyNodes(
	set<Node*> nodes_to_copy,
    unordered_map<Node*, Node*> argument_replacements,
	unordered_map<int, Node*> indices,
	unordered_set<Node*> targets, bool must_copy_all) {

	// if we have indices, we are basically rerunning the computation with a
	// different set of indices (of possible different shape)
	bool can_change_shape = !indices.empty();
	NodeArguments shape_args = NodeArguments();
	if (can_change_shape) {
		// get first index
		int first_index = indices.begin()->first;
		Node* first_index_node = indices.at(first_index);
		shape_args = first_index_node->args.GetArguments(ArgType::Shape);
	}

	if (nodes_to_copy.empty()) {
		return {};
	}

	if (nodes_to_copy.size() > 1024) {
		throw std::runtime_error(
		    "Copy Nodes: Copying too many nodes, something is probably "
		    "wrong. Number of nodes to copy: " +
		    to_string(nodes_to_copy.size()));
	}

	// copy the nodes
	map<Node*, Node*> copied_node_map;
	for (auto node = begin(); !node.end(); node.next()) {
		if (!nodes_to_copy.contains(node.get())) {
			continue;
		}

		Node* new_node;

		// if we have the index, use it instead
		bool is_dim = node->name == "dim_id";
		bool no_index = true;
		if (is_dim) {
			int dim = node->data[0];
			if (indices.contains(dim)) {
				new_node = indices.at(dim);
				no_index = false;
			}
		}

		if (no_index) {
			// create new arguments
			NodeArguments new_args;
			for (auto& [arg, from]: node->args.inputs_) {
				auto& [type, index] = arg;
				if (can_change_shape && type == ArgType::Shape) {
					continue;
				}

				// if shape or memory argument, then no need to use copied node
				if (node->args.CannotCopyArgument(arg) && !targets.contains(from) && !argument_replacements.contains(from)) {
					if(from == nullptr) {
						throw std::runtime_error("Copy Nodes: From is null for node " + node->name);
					}
					new_args[arg] = from;
					continue;
				}

				Node* new_from = from;

				if (argument_replacements.contains(from)) {
					new_from = argument_replacements[from];
				} else if (nodes_to_copy.contains(from)) {
					if(!copied_node_map.contains(from)) {
						throw std::runtime_error("Copy Nodes: No replacement for node " + from->name);
					}
					new_from = copied_node_map[from];
				} else if (must_copy_all) {
					throw std::runtime_error("Copy Nodes: No replacement for node " + from->name + " but we must copy all nodes");
				}

				if(new_from == nullptr) {
					throw std::runtime_error("Copy Nodes: New from is null for node " + from->name);
				}

				// create new argument
				new_args[arg] = new_from;
			}

			if (can_change_shape) {
				new_args.insert(shape_args.begin(), shape_args.end());
			}

			// create new node
			Tensor* tensor = Tensor::GetCopy(*node->GetTensor(), new_args);
			new_node = tensor->node_;
		}

		if(new_node == nullptr) {
			throw std::runtime_error("Copy Nodes: New node is null for node " + node->name);
		}

		copied_node_map[node.get()] = new_node;
	}

	return copied_node_map;
}

/// <summary>
/// Copy nodes together with their arguments (as far as possible)
/// </summary>
/// <param name="targets">nodes to copy</param>
/// <param name="indices">if given, the indices to use</param>
/// <returns>map between the original nodes and the copied nodes</returns>
map<Node*, Node*> IR::CopyComputation(
    const unordered_set<Node*>& targets, const unordered_map<int, Node*>& indices) {

	// do a depth first search to copy all the nodes required for the targets
	// (only if in the same kernel)
	set<Node*> nodes_to_copy;
	std::function<void(Node*)> dfs = [&](Node* node) {
		if (nodes_to_copy.contains(node)) return;
		nodes_to_copy.insert(node);
		for (auto& [arg, from] : node->args.inputs_) {
			if (node->args.CannotCopyArgument(arg)) continue;
			dfs(from);
		}
	};

	for (Node* target : targets) {
		dfs(target);
	}

	return CopyNodes(nodes_to_copy, {}, indices, targets, true);
}

map<Node*, Node*> IR::CopyNodesWithIndex(unordered_set<Node*> nodes_to_copy,
                                         unordered_map<int, Node*> indices,
                                         Node* cursor) {
	// copy all the nodes at the beginning of the kernel
	map<Node*, Node*> copied_node_map;
	ExecuteExpressionBefore(cursor, [&]() {
		copied_node_map = CopyComputation(nodes_to_copy, indices);
	});

	return copied_node_map;
}

void IR::CopyArguments(ArgEdges args_to_copy, Node* cursor)
{
	unordered_set<Node*> nodes_to_copy;
	for (auto& [arg, out] : args_to_copy) {
		nodes_to_copy.insert(arg.second);
	}

	// copy all the nodes at the beginning of the kernel
	map<Node*, Node*> copied_node_map;
	unordered_map<int, Node*> indices;
	copied_node_map = CopyNodesWithIndex(nodes_to_copy, indices, cursor);

	// replace all the arguments that use the copied nodes
	for (auto& [arg, out] : args_to_copy) {
		Node* from = arg.second;
		if (!copied_node_map.contains(from)) {
			throw std::runtime_error("Optimize Kernels: Copy Fail");
		}
		Node* to = copied_node_map[from];
		out->args.UpdateArgument(arg.first, to);
	}
}

#define MAX_KERNEL_COPY_COST 2048.0f

void IR::OptimizeKernels() {
	// get kernel data
	vector<Node*> kernels = GetNodesOfType("kernel");
	ComputeNodeCost();

	// go over each kernel and copy computations outside the kernel if they are
	// cheap enough
	for (auto kernel : kernels) {
		ArgEdges args_to_copy;
		ArgEdges shape_args_to_copy;
		// go over all nodes in the kernel and check if their inputs can be copied
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			// go over all inputs
			for (auto& [arg, from]: node->args.inputs_) {
				bool inside_kernel = from->HasParent(kernel);
				bool from_in_kernel = from->HasParent("kernel");

				if (!inside_kernel && !node->args.CannotCopyArgument(arg))
				{
					// check if input is cheap enough to copy
					float input_cost = from->cost_;
					if (input_cost == -1.0) {
						//throw std::runtime_error("Cost has not been computed for node " + input.from_->get()->var_name);
						continue;
					}
					bool cheap_enough = input_cost >= 0.0f && input_cost < MAX_KERNEL_COPY_COST;
					bool has_only_one_output = from->args.outputs_.size() == 1;
					if (cheap_enough || has_only_one_output) {
						args_to_copy.push_back(ArgEdge(Arg(arg, from), *node));
					}
				}
				//shape arguments can not be inside kernels
				if (from_in_kernel && arg.first == ArgType::Shape) {
					shape_args_to_copy.push_back(ArgEdge(Arg(arg, from), *node));
				}
			}
		}

		//go over kernel shape arguments
		for (int i = 0; i < kernel->args.Count(ArgType::Shape); i++) {
			Node* shape_node = kernel->args.Get(ArgType::Shape, i);
			bool from_in_kernel =shape_node->HasParent("kernel");
			if (from_in_kernel) {
				shape_args_to_copy.push_back(ArgEdge(Arg(ArgID(ArgType::Shape, i), shape_node), kernel));
			}
		}

		// copy the nodes that are outside the kernel inside
		CopyArguments(args_to_copy, kernel->child);
		// copy shape arguments before the kernel
		CopyArguments(shape_args_to_copy, kernel);
	}
}

#define MAX_LOAD_COPY 5000.0f
#define MAX_LOAD_COPY_COUNT 2
#define MAX_LOAD_SIZE_RATIO 0.5f
void IR::OptimizeKernelLoadOperations() {
	ComputeNodeCost();

	vector<Node*> kernels = GetNodesOfType("kernel");

	unordered_set<Node*> nodes_to_remove;

	for (auto kernel : kernels) {
		ShapeInfo kernel_shape = ShapeInfo(kernel);

		unordered_map<Node*, Node*> loads_to_copy;
		unordered_set<Node*> memory_inputs;
		// go over all nodes in the kernel and check if their inputs can be copied
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			if (node->name != "load") continue;

			//get memory input
			Node* memory_input = node->args.Get(ArgType::Memory);

			ShapeInfo memory_shape = ShapeInfo(memory_input);

			bool inside_kernel = memory_input->HasParent("kernel");
			if (!inside_kernel) continue;

			bool is_not_modified = !memory_input->HasFlags(NodeFlags::Modified);
			if (!is_not_modified) continue;

			float size_ratio = ShapeInfo::GetSizeRatio(kernel_shape, memory_shape);

			int output_count = (int)memory_input->args.outputs_.size();
			//only fuse if this is used less than MAX_LOAD_COPY_COUNT times or we can reduce dimensionality by fusing
			bool fusion_makes_sense = (output_count < MAX_LOAD_COPY_COUNT) ||
			                          (size_ratio <= MAX_LOAD_SIZE_RATIO);
			bool cheap_enough = memory_input->cost_ >= 0.0f &&
			                    memory_input->cost_ < (MAX_LOAD_COPY / output_count);

			//if the memory input is used only once and is not a memory node
			if (cheap_enough && fusion_makes_sense) {
				loads_to_copy[memory_input] = *node;
				memory_inputs.insert(memory_input);
			}
		}

		for (auto load : loads_to_copy) {
			//get the load
			Node* memory_input = load.first;
			Node* load_node = load.second;

			//get the indices
			unordered_map<int, Node*> indices;
			for (auto& [arg, from] : load_node->args.inputs_) {
				if (arg.first == ArgType::Index) {
					indices[arg.second] = from;
				}
			}

			//copy the load node
			map<Node*, Node*> copied_node_map = CopyNodesWithIndex({ memory_input }, indices, load_node);


			Tensors indices_tensors = Tensors();
			indices_tensors.resize(indices.size());
			for (auto& [index, node] : indices) {
				indices_tensors[index] = node->GetTensor();
			}

			//go over all the copied nodes and add load nodes to their inputs that are outside the kernel
			for (auto& [old_node, new_node] : copied_node_map) {
				AddNodeLoadOperations(new_node, kernel, indices_tensors);
			}

			Node* copied_load = copied_node_map[memory_input];
			//copy over the information from the original load node
			copied_load->CopyMetadata(load_node);

			//go over all outputs of the load node and replace them with the copied nodes
			for (auto [edge, to] : load_node->args.outputs_) {
				auto& [id, from] = edge;
				to->args.UpdateArgument(id, copied_load);
			}

			//remove the load node since it is not needed anymore
			nodes_to_remove.insert(load_node);
		}
	}

	// remove the load nodes
	for (auto node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();
}

#define MAX_HOST_COPY_COST 8192.0f

void IR::OptimizeHost() {
	ComputeNodeCost();

	//loop over all nodes and copy their arguments if they are cheap enough and inside kernels
	for (auto node = begin(); !node.end(); node.next()) {
		if (node->HasParent("kernel")) {
			continue;
		}

		ArgEdges args_to_copy;
		// go over all inputs
		for (auto& [arg, from] : node->args.inputs_) {
			bool inside_kernel = from->HasParent("kernel");

			if (inside_kernel && !node->args.CannotCopyArgument(arg)) {
				// check if input is cheap enough to copy
				float input_cost = from->cost_;
				if (input_cost == -1.0) {
					//throw std::runtime_error("Cost has not been computed for node " + input.from_->get()->var_name);
					continue;
				}
				bool cheap_enough = input_cost >= 0.0f && input_cost < MAX_HOST_COPY_COST;
				bool has_only_one_output = from->args.outputs_.size() == 1;

				if (cheap_enough || has_only_one_output) {
					args_to_copy.push_back(ArgEdge(Arg(arg, from), *node));
				} else {
					throw std::runtime_error("Host optimization: Copy cost too high for node " + node->name + " with cost " + to_string(input_cost));
				}
			}
		}

		CopyArguments(args_to_copy, node.get());
	}
}



void IR::MoveShapeOutsideKernels() {
	// find all nodes that are used as shapes and are inside kernels
	map<Node*, Node*> nodes_to_copy;
	for (auto node = begin(); !node.end(); node.next()) {
		Node* kernel = node->GetParent("kernel");
		if (kernel == *node) continue;

		// go over all outputs arguments
		for (auto [edge, to] : node->args.outputs_) {
			auto& [id, from] = edge;
			if (id.first != ArgType::Shape) {
				continue;
			}
			// add the node to the set
			nodes_to_copy[node.get()] = kernel;
		}
	}

	for (auto [ node, kernel ] : nodes_to_copy) {
		//get all output arguments that are shapes
		ArgEdges args_to_copy;
		int earliest_output_index = INT_MAX;
		Node* earliest_output = nullptr;
		for (auto [edge, to] : node->args.outputs_) {
			auto& [id, from] = edge;
			if (id.first == ArgType::Shape) {
				args_to_copy.push_back(ArgEdge(Arg(id, node), to));

				//get the earliest output
				if (to->index_ < earliest_output_index) { //wat
					earliest_output_index = to->index_;
					earliest_output = to;
				}
			}
		}

		Node* common_parent = earliest_output->GetCommonParent(kernel);

		// copy shape computation and put it before the earliest output (outside of the kernel if its inside)
		CopyArguments(args_to_copy, common_parent);
		UpdateGraph();
	}
}


/// <summary>
/// Get all inputs of this program in the IR
/// </summary>
void IR::GetInputList() {
	int input_memory_index = 0;
	//MUST BE IN ORDER
	for (auto node = begin(); !node.end(); node.next()) {
		if (node->memory_type_ == MemoryType::Input) {
			shape_memory_map[*node] = {};
			// add shapes to the memory inputs
			for (int i = 0; i < node->args.Count(ArgType::Shape); i++) {
				Node* shape_node = node->args.Get(ArgType::Shape, i);
				shape_memory_map[*node][i] = shape_node;
			}

			// set input memory index
			int input_index = input_memory_index++;
			// add shapes to the memory inputs
			input_memory_map[input_index] = *node;
			node->special_indices_[0] = input_index;
			//if any of the inputs are "input_shape" then we need to add the input index to them
			for (auto& [arg, from] : node->args.inputs_) {
				if (arg.first == ArgType::Shape && from->name == "input_shape") {
					if(!from->special_indices_.contains(1)) { //ONLY FIRST TIME
						from->special_indices_[1] = input_index;
					}
				}
			}
		}
	}
}

/// <summary>
/// Get all outputs of this program in the IR
/// </summary>
void IR::GetOutputList() {
	for (auto node = begin(); !node.end(); node.next()) {
		if (node->memory_type_ == MemoryType::Output) {
			if (!node->op->HasAllTypes(OpClass::Memory)) {
				throw std::runtime_error(
				    "Compilation error: output is not a memory node");  // all outputs
				                                                        // should be
				                                                        // memory nodes
				                                                        // at this point
			}
			output_memory_map[node->special_indices_[0]] = *node;
		}
		if (node->op->HasAllTypes(OpClass::Modifier, OpClass::MemoryOp)) {
			if (!node->HasParent("kernel")) {
				writebacks++;
			}
		} else if (node->op->HasAllTypes(OpClass::Load, OpClass::MemoryOp)) {
			if (!node->HasParent("kernel")) {
				readbacks++;
			}
		}
	}
}

/// <summary>
/// Compute statistics about the IR
/// </summary>
void IR::ComputeStatistics() {
	for (auto node = begin(); !node.end(); node.next()) {
		if (node->name == "memory") {
			if (node->memory_type_ == MemoryType::Output) {
				output_memory_count++;
			} else if (node->memory_type_ == MemoryType::Input) {
				input_memory_count++;
			} else {
				temp_memory_count++;
			}
		}
	}
}

bool isConstantAndEqualTo(const Tensor* tensor, float value) {
	if (tensor->node_->name != "const" || tensor->node_->HasFlags(NodeFlags::Modified)) {
		return false;
	}

	switch (tensor->node_->type) {
		case TFType::Float:
			return AsFloat(tensor->node_->data[0]) == value;
		case TFType::Int:
			return AsInt(tensor->node_->data[0]) == value;
		case TFType::Uint:
			return tensor->node_->data[0] == value;
		default:
			throw std::runtime_error("Unexpected type in isConstantAndEqualTo");
	}
}

bool isConstant(const Tensor* tensor) {
	return tensor->node_->name == "const" && !tensor->node_->HasFlags(NodeFlags::Modified);
}

Tensor* ApplyMultiOP(const Tensor* a, const Tensor* b, std::function<float(float, float)> opF32, std::function<int(int, int)> opI32, std::function<uint(uint, uint)> opU32) {
	switch (a->node_->type) {
		case TFType::Float:
			return &Tensor::Constant(opF32(AsFloat(a->node_->data[0]), AsFloat(b->node_->data[0])));
		case TFType::Int:
			return &Tensor::Constant(opI32(AsInt(a->node_->data[0]), AsInt(b->node_->data[0])));
		case TFType::Uint:
			return &Tensor::Constant(opU32(a->node_->data[0], b->node_->data[0]));
		default:
			throw std::runtime_error("Unexpected type in ApplyMultiOP");
	}
}

Tensor* ApplyUnaryOP(const Tensor* a, std::function<float(float)> opF32, std::function<int(int)> opI32, std::function<uint(uint)> opU32) {
	switch (a->node_->type) {
		case TFType::Float:
			return &Tensor::Constant(opF32(AsFloat(a->node_->data[0])));
		case TFType::Int:
			return &Tensor::Constant(opI32(AsInt(a->node_->data[0])));
		case TFType::Uint:
			return &Tensor::Constant(opU32(a->node_->data[0]));
		default:
			throw std::runtime_error("Unexpected type in ApplyUnaryOP");
	}
}

#define ApplyOP(v1, v2, op) ApplyMultiOP(v1, v2, [](float a, float b) { return a op b; }, [](int a, int b) { return a op b; }, [](uint a, uint b) { return a op b; })
#define ApplyFUNC(v1, v2, func) ApplyMultiOP(v1, v2, [](float a, float b) { return func(a, b); }, [](int a, int b) { return func(a, b); }, [](uint a, uint b) { return func(a, b); })
#define ApplyUOP(v1, op) ApplyUnaryOP(v1, [](float a) { return op a; }, [](int a) { return op a; }, [](uint a) { return op a; })
#define ApplyUFUNC(v1, func) ApplyUnaryOP(v1, [](float a) { return func(a); }, [](int a) { return func(a); }, [](uint a) { return func(a); }

void IR::OptimizeOperations()
{
	for (auto node = begin(); !node.end(); node.next()) {
		//get node operation
		const string op = node->name;

		//get inputs
		map<int, const Tensor*> inputs = node->args.GetTensors(ArgType::Input);
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
					result = &Tensor::Constant(0u, inputs[0]->node_->type);
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
					result = &Tensor::Constant(0u, inputs[0]->node_->type);
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
			else if (op == "clamp") {
				// if all are constants, replace with result
				if (isConstant(inputs[0]) && isConstant(inputs[1]) && isConstant(inputs[2])) {
					// compute result
					result = ApplyFUNC(inputs[0], inputs[1], max);
					result = ApplyFUNC(result, inputs[2], min);
				}
			}
			else if(op == "neg") {
				if(isConstant(inputs[0])) {
					result = ApplyUnaryOP(inputs[0], [](float a) { return -a; }, [](int a) { return -a; }, [](uint a) { return a; });
				}
			}
			//TODO (Moroz): add more optimizations

			// if computed optimized result, replace all node references with it
			if (result != nullptr)
			{
				node->MakeOutputsUseGivenNode(result->node_);
			}
		});
	}
}

unordered_set<Node*> IR::GetDependencies(unordered_set<Node*> nodes) {
	unordered_set<Node*> dependencies;
	std::function<void(Node*)> dfs = [&](Node* node)
	{
		if (dependencies.contains(node)) {
			return;
		}

		dependencies.insert(node);

		//all inputs of this node are used
		for (auto& [arg, from] : node->args.inputs_) {
			dfs(from);
		}

		//if the node is a memory node or used as memory, then all outputs are used
		for (auto [edge, to] : node->args.outputs_) {
			auto& [id, from] = edge;
			if (to->args.IsChangingInput(id)) {
				dfs(to);
			}
		}
	};

	for(auto node : nodes) {
		dfs(node);
	}

	return dependencies;
}

void IR::RemoveUnusedOperations() {
	unordered_set<Node*> used_nodes;
	//mark all output nodes as used
	for (auto node = begin(); !node.end(); node.next()) {
		if (node->memory_type_ == MemoryType::Output ||
		    node->memory_type_ == MemoryType::Input ||
		    node->op->HasAllTypes(OpClass::Static)) {
			used_nodes.insert(node.get());
		}
	}

	used_nodes = GetDependencies(used_nodes);

	// remove all nodes that are not used
	unordered_set<Node*> nodes_to_remove;
	for (auto node = begin(); !node.end(); node.next()) {
		if (!used_nodes.contains(node.get())) {
			if (node->memory_type_ != MemoryType::Input && node->memory_type_ != MemoryType::Output)
			{
				nodes_to_remove.insert(node.get());
			}
		}
	}

	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();
}

void IR::ComputeNodeCost()
{
	for (auto node = begin(); !node.end(); node.next()) {
		bool is_memory = node->op->HasAllTypes(OpClass::Memory);
		unordered_map<Node*, float> input_costs;
		for (auto& [id, from] : node->args.inputs_) {
			if (id.first != ArgType::Memory &&
			    (id.first != ArgType::Shape && !is_memory)) {
				input_costs[from] = from->cost_;
			}
		}
		float input_cost = node->op->GetCost();
		for (auto& input : input_costs) {
			input_cost += abs(input.second);
		}
		node->cost_ = input_cost;
	}
}

map<Node *, ArgEdges> IR::GetKernelOutputs(Node *kernel)
{
	UpdateGraph();
	map<Node*, ArgEdges> node_output;
	for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
		bool is_output = node->memory_type_ == MemoryType::Output;
		ArgEdges outputs = ArgEdges();

		for (auto [edge, to] : node->args.outputs_) {
			auto& [id, from] = edge;
			if (to == nullptr) continue;
			// if is a shape or memory argument, then skip (shape is loaded on CPU)
			if (id.first == ArgType::Shape) continue;
			if (!to->HasParent(kernel)) {
				outputs.emplace_back(Arg(id, *node), to);
				is_output = true;
			}
		}

		if (is_output) {
			node_output[*node] = outputs;
		}
	}

	return node_output;
}

void IR::AddNodeLoadOperations(Node* node, Node* kernel, Tensors indices) {
	for (auto& [arg, input_node] : node->args.inputs_) {
		if (arg.first == ArgType::Memory || arg.first == ArgType::Shape)
			continue;

		bool is_in_a_kernel = input_node->HasParent("kernel");
		bool is_outside = !input_node->HasParent(kernel);
		bool is_memory = input_node->op->HasAllTypes(OpClass::Memory);

		if (is_memory || (is_in_a_kernel && is_outside)) {
			// load the memory node before this node
			ExecuteExpressionBefore(node, [&]() {
				Tensor& loaded = Tensor::Load(*input_node->GetTensor(), indices, true);
				node->args.UpdateArgument(arg, loaded.node_);
			});
		}
	}
}

void IR::AddKernelGlobalLoadOperations() {
	// get kernels
	vector<Node*> kernels = GetNodesOfType("kernel");
	for (auto kernel : kernels) {

		// replace all inputs pointing to memory nodes with the memory node
		unordered_set<Node*> nodes_to_load;
		unordered_map<Node*, ArgEdges> load_arguments;
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			for (auto& [arg, input_node] : node->args.inputs_) {
				if (arg.first == ArgType::Memory || arg.first == ArgType::Shape)
					continue;

				bool is_in_a_kernel = input_node->HasParent("kernel");
				bool is_outside = !input_node->HasParent(kernel);
				bool is_memory = input_node->op->HasAllTypes(OpClass::Memory);

				if (is_memory || (is_in_a_kernel && is_outside)) {
					nodes_to_load.insert(input_node);
					load_arguments[input_node].push_back(ArgEdge(Arg(arg, input_node), node.get()));
				}
			}
		}

		for (auto node : nodes_to_load) {
			// load the memory node at the beginning of the kernel
			ExecuteExpressionFirstChild(kernel, [&]() {
				Tensor& loaded = Tensor::Load(*node->GetTensor(), {}, true);
				for (auto [in, out] : load_arguments[node]) {
					auto& [arg, from] = in;
					out->args.UpdateArgument(arg, loaded.node_);
				}
			});
		}

		UpdateGraph();
	}
}


void IR::AddMemoryOpIndices() {
	// get kernels
	vector<Node*> kernels = GetNodesOfType("kernel");
	for (auto kernel : kernels) {
		// get kernel shape arguments
		NodeArguments shape_args = kernel->args.GetArguments(ArgType::Shape);

		Tensors indices = Tensors();
		// add dimension index nodes
		ExecuteExpressionFirstChild(kernel, [&]() {
			for (int i = 0; i < shape_args.size(); i++) {
				indices.push_back(&Tensor::Index(shape_args, i));
			}
		});
		int kernel_dim = (int)shape_args.size();

		// replace all inputs pointing to memory nodes with the memory node
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			if (!node->op->HasAllTypes(OpClass::MemoryOp)) {
				continue;
			}

			Node* input_node = node->args.Get(ArgType::Memory);
			map<int, const Tensor*> shape = input_node->args.GetTensors(ArgType::Shape);

			int memory_dim = (int)shape.size();
			ExecuteExpressionBefore(node.get(), [&]() {
				for (int i = 0; i < memory_dim; i++) {
					if (node->args.Has(ArgType::Index,i)) {
						continue;
					}
					if (memory_dim > kernel_dim) {
						throw std::runtime_error(
						    "Memory dimension is greater than kernel dimension, we can't "
						    "implicitly broadcast");
					}
					const Tensor* index = nullptr;
					// if the shape is 1, then we broadcast and set the index to 0
					if (isConstantAndEqualTo(shape[i], 1.0)) {
						index = &Tensor::Constant(0);
					} else {
						index = indices[kernel_dim - memory_dim + i];
					}
					node->args.AddArgument(ArgType::Index, i, index->node_);
				}
			});
		}
	}

	UpdateGraph();
}

void IR::AddKernelGlobalStoreOperations() {
	// get kernels
	vector<Node*> kernels = GetNodesOfType("kernel");

	// go over all outputs of each kernel and create memory nodes to store the
	// output
	for (auto kernel: kernels) {
		map<Node*, ArgEdges> node_output = GetKernelOutputs(kernel);

		for (auto [output, args] : node_output) {
			// if the output is already a memory node, then skip
			if (output->op->HasAllTypes(OpClass::Memory)) {
				continue;
			}

			Node* mem;
			// add memory node before this kernel
			ExecuteExpressionBefore(kernel, [&]() {
				mem = Tensor::Memory(kernel->args.GetArguments(ArgType::Shape), output->type).node_;
				mem->debug_name = output->debug_name;

				if (output->memory_type_ == MemoryType::Output) {
					mem->memory_type_ = MemoryType::Output;
					mem->special_indices_ = output->special_indices_;
					output->memory_type_ = MemoryType::None;
				}
			});

			// go over all outputs of this node and replace their input with the
			// memory node
			for (auto& [arg, to] : args) {
				auto id = arg.first;
				if (id.first != ArgType::Shape &&
				    id.first != ArgType::Memory) {
					// if not a memory or shape argument, then the memory needs to be
					// loaded before the node
					ExecuteExpressionBefore(to, [&]() {
						Tensor& loaded = Tensor::Load(*mem->GetTensor(), {}, true);
						// the node must now use the loaded value
						to->args.UpdateArgument(id, loaded.node_);
					});
				} else {
					// otherwise the memory can be used directly
					to->args.UpdateArgument(id, mem);
				}
			}

			//get last modification of the memory
			Node* last_mod = output->GetFinalVersion();
			//get the parent of the last modification on the same level as the memory
			Node* last_mod_parent = last_mod->GetCommonParent(output);

			// add store node after the last modification on the same level as the memory
			ExecuteExpressionAfter(last_mod_parent, [&]() {
				// add store node after this node
				Tensor* store = &Tensor::Store(*mem->GetTensor(), *output->GetTensor(), {}, true);
			});
		}
	}

	// replace all inputs pointing to memory nodes with the memory node
	for (auto node = begin(); !node.end(); node.next()) {
		bool is_memory = node->op->HasAllTypes(OpClass::Memory);

		for (auto& [id, from] : node->args.inputs_) {
			if (id.first == ArgType::Memory ||
			    (id.first  == ArgType::Shape && !is_memory))
				continue;

			if (from->op->HasAllTypes(OpClass::Memory)) {
				// load the memory node before this node
				ExecuteExpressionBefore(node.get(), [&]() {
					Tensor& loaded = Tensor::Load(*from->GetTensor(), {}, true);
					node->args.UpdateArgument(id, loaded.node_);
				});
			}
		}
	}

	UpdateGraph();
}

void IR::AddMemoryDeallocation()
{
	vector<Node*> memory_nodes = GetNodesOfType("memory");

	// go over all outputs of each memory and and put a deallocation node after the last time it is used
	for (auto memory : memory_nodes) {
		// skip input and output memories, they are deallocated manually
		if (memory->memory_type_ == MemoryType::Input) {
			continue;
		}

		Node* last_output = nullptr;
		int last_output_index = -1;

		bool is_an_output = false;

		//do a dfs to find the last output
		std::function<void(Node*)> dfs = [&](Node* node) {
			if (node->memory_type_ == MemoryType::Output) {
				is_an_output = true;
				return;
			}

			for (auto [edge, to] : node->args.outputs_) {
				auto& [id, from] = edge;
				if (to->op->HasAllTypes(OpClass::MemoryReuse)) {
					dfs(to);
				} else {
					if (last_output_index < to->index_) {
						last_output_index = to->index_;
						last_output = to;
					}
				}
			}
		};

		dfs(memory);

		if (is_an_output) {
			continue;
		}

		// need to add deallication in the same scope as the allocation
		Node* deallocation_point = last_output->GetCommonParent(memory);

		// add deallocation node after the last time the memory is used
		ExecuteExpressionAfter(deallocation_point, [&]() {
			Tensor* deallocate = &Tensor::Deallocate(*memory->GetTensor());
		});
	}

	UpdateGraph();
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
Tensor* ComputeFlatIndex(NodeArguments memory_shape, vector<Tensor*> indices, map<int, const Tensor*> idx, int memory_dim, TensorIndexingMode mode = TensorIndexingMode::Clamp)
{
	if (memory_dim == 0)
	{
		return &Tensor::Constant(0);
	}

	int kernel_dim = (int)indices.size();

	function<const Tensor*(int)> get_shape = [&](int dim) {
		return memory_shape[ArgID(ArgType::Shape, dim)]->GetTensor();
	};

	// function to get index for given dimension, if not found then return
	// default dim index
	function<Tensor*(int)> get_index = [&](int dim) {
		Tensor* out;
		if (idx.find(dim) != idx.end()) {
			out = const_cast<Tensor*>(idx[dim]);
		} else {
			throw std::runtime_error("Finalize memory indexing: node index not found");
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
				throw std::runtime_error("Finalize memory indexing: invalid tensor indexing mode");
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

void IR::ReplaceDimNodes(Node* kernel, vector<Tensor*> indices, int dims)
{
	// replace all dim nodes with the corresponding index node
	unordered_set<Node*> nodes_to_remove;
	for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
		if (node->name == "dim_id") {
			int dim = node->data[0];
			if (dim >= dims) {
				throw runtime_error("Invalid dimension index " + to_string(dim) +
													" for kernel of size " + to_string(dims));
			}

			// remove the dim node
			nodes_to_remove.insert(node.get());
		}
		else
		{
			//go over node inputs and replace dim nodes with index nodes
			for (auto& [id, from] : node->args.inputs_) {
				if (from->name == "dim_id") {
					int dim = from->data[0];
					if (dim >= dims) {
						throw runtime_error("Invalid dimension index " + to_string(dim) +
																							" for kernel of size " + to_string(dims));
					}

					// replace the dim node with the index node
					node->args.UpdateArgument(id, indices[dim]->node_);
				}
			}
		}
	}

	UpdateGraph();

	// remove all dim nodes
	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();
}

void IR::MultiDimensionalModeIndices(vector<Tensor*>& indices, Node* kernel_, int dims, Tensors kernel_shape)
{
	//add dim_id nodes at the beginning of the kernel
	ExecuteExpressionFirstChild(kernel_, [&]() {
		for (int i = 0; i < dims; i++) {
			indices[i] = &Tensor::Index(kernel_shape, i);
		}
	});
}

vector<Tensor*> ComputeIndicesFromBlockIndex(Tensor* block_index, Node* kernel,
                                             Tensors kernel_shape, int dims) {
	//compute in-block index
	vector<int> block_size = kernel->group_size;
	int block_dim = (int)block_size.size();
	Tensors block_size_tensors = {};
	for (int i = 0; i < block_dim; i++) {
		block_size_tensors.push_back(&Tensor::Constant(block_size[i]));
		//block_size_tensors[i]->SetDebugName("block_size_" + to_string(i));
	}
	vector<Tensor*> in_block_indices;
	for (int i = 0; i < block_dim; i++) {
		in_block_indices.push_back(&block_index->BlockThreadIndex(block_dim - 1 - i));
	}

	//compute out-of-block index
	Tensors blocks_shape = {};
	for (int i = 0; i < dims - block_dim; i++) {
		blocks_shape.push_back(kernel_shape[i]);
		blocks_shape[i]->SetDebugName("blocks_shape_" + to_string(i));
	}
	//the rest are divided into blocks of the given size
	for (int i = 0; i < block_dim; i++) {
		const Tensor block_size = *block_size_tensors[i];
		const Tensor shape = *kernel_shape[dims - block_dim + i];
		Tensor& ceil = (shape + block_size - Tensor::Constant(1)) / block_size;
		blocks_shape.push_back(&ceil);
		blocks_shape[dims - block_dim + i]->SetDebugName("blocks_shape_" + to_string(dims - block_dim + i));
	}
	vector<Tensor*> out_block_indices = ComputeIndicesFromLinearIndex(block_index, blocks_shape, dims);
	for (int i = 0; i < dims; i++) {
		//out_block_indices[i]->SetDebugName("out_block_index_" + to_string(i));
	}

	//combine the indices
	vector<Tensor*> indices = {};
	for (int i = 0; i < dims - block_dim; i++) {
		indices.push_back(out_block_indices[i]);
		indices[i]->SetDebugName("index_" + to_string(i));
	}
	//the rest are sum of block index and in-block index
	for (int i = 0; i < block_dim; i++) {
		indices.push_back(&(*out_block_indices[dims - block_dim + i] * *block_size_tensors[i] + *in_block_indices[i]));
		indices[dims - block_dim + i]->SetDebugName("index_" + to_string(dims - block_dim + i));
	}

	return indices;
}

Tensor* IR::LinearBlockModeIndices(vector<Tensor*>& indices, Node* kernel_, int dims, Tensors kernel_shape)
{
	Tensor* block_index = nullptr;
	Tensor* if_tensor = nullptr;
	ExecuteExpressionFirstChild(kernel_, [&]() {
		block_index = &kernel_->GetTensor()->BlockIndex();

		switch (dims)
		{
			case 1:
				kernel_->group_size = {256};
				break;
			case 2:
				kernel_->group_size = {16, 16};
				break;
			case 3:
				kernel_->group_size = {8, 8, 8};
				break;
			default:
				kernel_->group_size = {8, 8, 8};
		}

		//if the dimensions are known, then use the minimum of the group size and the shape to avoid useless computation
		int group_dim = (int)kernel_->group_size.size();
		for (int i = 0; i < group_dim; i++) {
			int shape = kernel_shape[dims - group_dim + i]->TryGetConstant();
			if (shape > 0) {
				kernel_->group_size[i] = min(kernel_->group_size[i], shape);
			}
		}

		indices = ComputeIndicesFromBlockIndex(block_index, kernel_, kernel_shape, dims);

		//add a check for if inside the dispatch
		Tensor* inside_dispatch = &(*indices[0] < *kernel_shape[0]);
		for (int i = 1; i < dims; i++) {
			inside_dispatch = &(*inside_dispatch && *indices[i] < *kernel_shape[i]);
		}
		inside_dispatch->SetDebugName("is_inside_dispatch");

		//put an if condition
		if_tensor = &Tensor::If(*inside_dispatch);
		if_tensor->SetDebugName("if_inside_dispatch");
	});

	ReplaceDimNodes(kernel_, indices, dims);

	return if_tensor;
}

void ComputeAddress(Node* node, vector<Tensor*> indices)
{
	// get the input memory node
	const Tensor* memory = node->args.GetTensor(ArgType::Memory);

	NodeArguments memory_shape = memory->node_->args.GetArguments(ArgType::Shape);

	int memory_dim = (int)memory_shape.size();

	// get the index nodes
	map<int, const Tensor*> idx = node->args.GetTensors(ArgType::Index);

	if (idx.empty())
	{
		node->indexing_mode_ = TensorIndexingMode::Unsafe; //we can guarantee that the index is in bounds
	}

	Tensor* flat_index = ComputeFlatIndex(memory_shape, indices, idx, memory_dim, node->indexing_mode_);

	// TODO(Moroz): add different modes for clamping (e.g. clamp, wrap,
	// mirror, zero)

	// remove the index node edges
	node->args.RemoveArguments(ArgType::Index);

	// add the flat index node edge
	node->args.AddArgument(ArgType::Index, 0, flat_index->node_);
}

void IR::FinalizeMemoryIndexing() {
	vector<Node*> kernels = GetNodesOfType("kernel");

	vector<Tensor*> dispatch_checks;

	for (auto kernel : kernels) {
		Node* shape_node = kernel;
		if (shape_node == nullptr) continue;
		// load kernel shape
		map<int, const Tensor*> kernel_shape_map = shape_node->args.GetTensors(ArgType::Shape);
		Tensors kernel_shape;
		for (auto& shape : kernel_shape_map) {
			kernel_shape.push_back(shape.second);
		}

		if (kernel_shape.empty()) {
			// can skip if no kernel shape - no index
			continue;
		}

		// compute the index for each dimension
		int dims = (int)kernel_shape.size();
		vector<Tensor*> indices = vector<Tensor*>(dims);
		dispatch_checks.push_back(
		    LinearBlockModeIndices(indices, kernel, dims, kernel_shape));

		// go over all nodes that take an index as input (e.g. load, store, atomic)
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			if (node->op->HasAllTypes(OpClass::MemoryOp)) {
				ExecuteExpressionBefore(*node, [&]() { ComputeAddress(node.get(), indices); });
			}
		}
	}

	//now compute address for all nodes that are not in a kernel
	for (auto node = begin(); !node.end(); node.next()) {
		if (!node->HasParent("kernel") && node->op->HasAllTypes(OpClass::MemoryOp)) {
			ExecuteExpressionBefore(node.get(), [&]() {
				vector<Tensor*> indices = {};
				ComputeAddress(node.get(), indices);
			});
		}
	}

	for (auto check : dispatch_checks) {
		// put the rest of the kernel starting from if_node->next_ as a child
		// MoveNodeTo(if_node->node_->child, if_node->node_->next);
		Node* if_node = check->node_;
		if_node->child = if_node->next;
		if_node->next->prev = nullptr;
		if_node->next->parent = if_node;
		if_node->next = nullptr;
	}

	UpdateGraph();
}

void IR::RemoveUnusedKernels()
{
	vector<Node*> kernels = GetNodesOfType("kernel");
	vector<Node*> nodes_to_remove;

	for (auto kernel : kernels) {
		// remove all kernel nodes that dont do anything
		int memory_modifiers = 0;
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			if (node->op->HasAllTypes(OpClass::Modifier, OpClass::MemoryOp)) {
				memory_modifiers++;
			}
			//if any output is outside the kernel, then the kernel is needed
			for (auto [edge, to] : node->args.outputs_) {
				auto& [id, from] = edge;
				if (!to->HasParent(kernel)) {
					memory_modifiers++;
				}
			}
		}
		if (memory_modifiers == 0) nodes_to_remove.push_back(kernel);
	}

	// remove all nodes that are not used
	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();
}

int GetAxis(int dims, int axis)
{
	if (axis < 0)
	{
		axis = dims + axis;
	}
	return axis;
}

Tensor* ComputeReduction(const Tensor* array, int axis,
                         std::function<Tensor*(Tensor*, Tensor*)> reduction_op, string debug_name = "",
                         uint initial = 0,
                         std::function<Tensor*(Tensor*)> element_op = nullptr) {
	// Get shape of the array
	Tensors shape = array->GetShape();

	axis = GetAxis((int)shape.size(), axis);

	// Get the number of dimensions
	int dims = (int)shape.size();

	Tensors sum_shape = Tensors();
	for (int i = 0; i < dims; i++) {
		if (i == axis) {
			continue;
		}
		sum_shape.push_back(shape[i]);
	}

	// get indices for all dimensions but the last
	Tensors indices = Tensors();
	for (int i = 0; i < dims - 1; i++) {
		indices.push_back(&Tensor::Index(sum_shape, i));
	}

	// if no dimensions, then add constant 1
	if (sum_shape.empty()) {
		sum_shape.push_back(&Tensor::Constant(1));
	}

	Tensors load_index = Tensors();
	for (int id = 0, d = 0; d < dims; d++) {
		if (d == axis) {
			load_index.push_back(&Tensor::Constant(sum_shape, 0));
		} else {
			load_index.push_back(indices[id++]);
		}
	}

	// start with the first value
	Tensor* reduced = &Tensor::Constant(sum_shape, initial, array->node_->type);
	reduced->SetDebugName(debug_name);

	// create a loop over the last dimension starting from the second value
	Tensor::Loop(Tensor::Constant(0), *shape[axis], Tensor::Constant(1),
	[&](const Tensor& i) {
		load_index[axis] = &i;
		
		// load the value
		Tensor* value = &Tensor::Load(*array, load_index, true);

		if (element_op != nullptr) {
			value = element_op(value);
		}

		reduced->Set(*reduction_op(reduced, value));
	});

	return reduced;
}

Tensor* ComputeScan(const Tensor* array, int axis, std::function<Tensor*(Tensor*, Tensor*)> scan_op, string debug_name = "", uint initial = 0) {
	// Get shape of the array
	Tensors shape = array->GetShape();

	Tensor* scan_result = &Tensor::Memory(shape, array->node_->type);

	axis = GetAxis((int)shape.size(), axis);

	// Get the number of dimensions
	int dims = (int)shape.size();

	Tensors sum_shape = Tensors();
	for (int i = 0; i < dims; i++) {
		if (i == axis) {
			continue;
		}
		sum_shape.push_back(shape[i]);
	}

	// get indices for all dimensions but the last
	Tensors indices = Tensors();
	for (int i = 0; i < dims - 1; i++) {
		indices.push_back(&Tensor::Index(sum_shape, i));
	}

	// if no dimensions, then add constant 1
	if (sum_shape.empty()) {
		sum_shape.push_back(&Tensor::Constant(1));
	}

	Tensors load_index = Tensors();
	for (int id = 0, d = 0; d < dims; d++) {
		if (d == axis) {
			load_index.push_back(&Tensor::Constant(sum_shape, 0));
		} else {
			load_index.push_back(indices[id++]);
		}
	}

	// start with the first value
	Tensor* reduced = &Tensor::Constant(sum_shape, initial, array->node_->type);
	reduced->SetDebugName(debug_name);

	// create a loop over the last dimension starting from the second value
	Tensor::Loop(Tensor::Constant(0), *shape[axis], Tensor::Constant(1),
	[&](const Tensor& i) {
		load_index[axis] = &i;
		// load the value
		Tensor* value = &Tensor::Load(*array, load_index, true);
		reduced->Set(*scan_op(reduced, value));
		Tensor::Store(*scan_result, *reduced, load_index, true);
	});

	return scan_result;
}

Tensor* ComputeSum(const Tensor* array, int axis) {
	return ComputeReduction(array, axis, [](Tensor* a, Tensor* b) {
	return &(*a + *b); }, "sum");
}

Tensor* ComputeNorm(const Tensor* array, int axis) { 
	return &Tensor::sqrt(Tensor::tofloat(*ComputeReduction(array, axis, 
		[](Tensor* a, Tensor* b) { return &(*a + *b); }, "norm", 0, 
		[](Tensor* a) { return &(*a * *a); })));
}

Tensor* ComputeMean(const Tensor* array, int axis) {
	Tensor* sum = ComputeSum(array, axis);
	Tensors shape = array->GetShape();
	axis = GetAxis((int)shape.size(), axis);
	return &(Tensor::tofloat(*sum) / Tensor::tofloat(*shape[axis]));
}

Tensor* ComputeMax(const Tensor* array, int axis) {
	uint initial = 0;
	if (array->node_->type == TFType::Float) {
		float init = -FLT_MAX;
		initial = *(uint*)&init;
	}
	else if (array->node_->type == TFType::Int) {
		int init = INT_MIN;
		initial = *(uint*)&init;
	}
	return ComputeReduction(
	    array, axis, [](Tensor* a, Tensor* b) { return &Tensor::max(*a, *b); },
	    "max", initial);
}

Tensor* ComputeMin(const Tensor* array, int axis) {
	uint initial = UINT_MAX;
	if (array->node_->type == TFType::Float) {
		float init = FLT_MAX;
		initial = *(uint*)&init;
	}
	else if (array->node_->type == TFType::Int) {
		int init = INT_MAX;
		initial = *(uint*)&init;
	}
	return ComputeReduction(
	    array, axis, [](Tensor* a, Tensor* b) { return &Tensor::min(*a, *b); },
	    "min", initial);
}

Tensor* ComputeProduct(const Tensor* array, int axis) {
	uint initial = 1;
	if (array->node_->type == TFType::Float) {
		float init = 1.0f;
		initial = *(uint*)&init;
	}
	return ComputeReduction(
	    array, axis, [](Tensor* a, Tensor* b) { return &(*a * *b); }, "prod", initial);
}

Tensor* ComputeAny(const Tensor* array, int axis) {
	return ComputeReduction(array, axis, [](Tensor* a, Tensor* b) { return &(*a || *b); }, "any", 0);
}

Tensor* ComputeAll(const Tensor* array, int axis) {
	return ComputeReduction(
	    array, axis, [](Tensor* a, Tensor* b) { return &(*a && *b); }, "all", ~0);
}

Tensor* ComputePrefixSum(const Tensor* array, int axis) {
	return ComputeScan(array, axis, [](Tensor* a, Tensor* b) { return &(*a + *b); }, "prefix_sum");
}

Tensor* Transpose(const Tensor* array, map<int, int> permutation) {
	ShapeInfo shapeinfo = array->GetShapeInfo();
	int old_dim = shapeinfo.dim;
	Tensors perm_shape = Tensors();
	int permuted_dim = (int)permutation.size();

	shapeinfo.ExpandDimensions(permuted_dim);
	Tensors shape = shapeinfo.GetTensors();

	for (int i = 0; i < permuted_dim; i++) {
		perm_shape.push_back(shape[permutation[i]]);
	}

	//create indices
	Tensors indices = Tensors();
	for (int i = 0; i < permuted_dim; i++) {
		indices.push_back(&Tensor::Index(perm_shape, i));
	}
	//permute indices to load the values
	Tensors perm_indices = Tensors(old_dim, nullptr);
	for (int i = 0; i < permuted_dim; i++) {
		int old = permutation[i] - std::max(permuted_dim - old_dim, 0);
		if(old >= 0) {
			perm_indices[old] = indices[i];
		}
	}
	//if any nullptr, then put a constant 0
	for (int i = 0; i < old_dim; i++) {
		if(perm_indices[i] == nullptr) {
			perm_indices[i] = &Tensor::Constant(0);
		}
	}

	Tensor& loaded = Tensor::Load(*array, perm_indices, true);
	loaded.SetDebugName("transposed");
	return &loaded;
}

Tensor* ReverseDim(const Tensor* array, int axis) {
	ShapeInfo shapeinfo = array->GetShapeInfo();
	int dims = shapeinfo.dim;
	Tensors shape = shapeinfo.GetTensors();
	Tensors indices = Tensors();
	for (int i = 0; i < dims; i++) {
		if (i == axis) {
			indices.push_back(&(*shape[i] - Tensor::Constant(1) - Tensor::Index(shape, i)));
		} else {
			indices.push_back(&Tensor::Index(shape, i));
		}
	}
	Tensor& loaded = Tensor::Load(*array, indices, true);
	loaded.SetDebugName("reversed");
	return &loaded;
}

Tensor* ComputeDot(const Tensor* a, const Tensor* b, int axis) {
	Tensors shape_a = a->GetShape();
	Tensors shape_b = b->GetShape();
	axis = GetAxis((int)shape_a.size(), axis);
	return ComputeSum(&(*a * *b), axis);
}

//compute the matrix multiplication of two last dimensions
//takes two tensors [T1, T2, ..., Tn, M, N] and [Tm, .., Tn, N, K] and returns [T1, T2, ..., Tm, M, K]
Tensor* ComputeMatMul(const Tensor* a, const Tensor* b) {
	ShapeInfo shape_a = a->GetShapeInfo();
	ShapeInfo shape_b = b->GetShapeInfo();

	if (shape_a.dim < 2 && shape_b.dim < 2) {
		throw std::runtime_error("Matrix multiplication requires at least one 2D tensor");
	}

	if(shape_a.dim < 2) {
		shape_a.ExpandDimensions(2);
	}
	if(shape_b.dim < 2) {
		shape_b.ExpandDimensions(2);
	}

	Tensors shape_a_tensors = shape_a.GetTensors();
	Tensors shape_b_tensors = shape_b.GetTensors();

	//get shape of the result
	Tensors shape_c = Tensors();
	int dim_a = shape_a.dim;
	int dim_b = shape_b.dim;
	int max_dim = 0;
	Tensors max_shape = Tensors();
	//get the shape with most dimensions
	if (dim_a < dim_b) {
		max_dim = dim_b;
		max_shape = shape_b_tensors;
	} else {
		max_dim = dim_a;
		max_shape = shape_a_tensors;
	}

	for (int i = 0; i < max_dim - 2; i++) {
		shape_c.push_back(max_shape[i]);
	}
	shape_c.push_back(shape_a_tensors[dim_a - 2]);
	shape_c.push_back(shape_b_tensors[dim_b - 1]);

	ShapeDimCompareResult result = CompareShapeDim(shape_a_tensors[dim_a - 1]->node_, shape_b_tensors[dim_b - 2]->node_);
	if (!result.compatible) {
		throw std::runtime_error("Inner dimensions of the matrices must match");
	}

	const Tensor* sum_shape = result.broadcast_dim->GetTensor();

	// get indices for c elements
	Tensors indices_c = Tensors();
	for (int i = 0; i < max_dim; i++) {
		indices_c.push_back(&Tensor::Index(shape_c, i));
	}

	// start with 0
	Tensor* c = &Tensor::Constant(shape_c, 0, a->node_->type);
	c->SetDebugName("matmul");

	// loop over k and compute += A t1t2..tN ik * B t1t2..tN kj
	Tensor::Loop(Tensor::Constant(0), *sum_shape, Tensor::Constant(1),
		[&](const Tensor& k) {

		// get indices for a elements
		Tensors indices_a = Tensors();
		for (int i = 0; i < dim_a - 2; i++) {
			indices_a.push_back(indices_c[max_dim - dim_a + i]);
		}
		indices_a.push_back(indices_c[max_dim - 2]);
		indices_a.push_back(&k);

		// get indices for b elements
		Tensors indices_b = Tensors();
		for (int i = 0; i < dim_b - 2; i++) {
			indices_b.push_back(indices_c[max_dim - dim_b + i]);
		}
		indices_b.push_back(&k);
		indices_b.push_back(indices_c[max_dim - 1]);

		// load the value
		Tensor* value = &(Tensor::Load(*a, indices_a, true) *
		                  Tensor::Load(*b, indices_b, true));

		c->Set(*c + *value);
	});
	
	return c;
}

void IR::InsertAlgorithmicPrimitives() {
	// get all nodes for each type
	vector<Node*> nodes = GetNodesOfType(OpClass::Algorithm);

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

			Tensor* result;
			if (node->name == "dim_sum") {
				result = ComputeSum(inputs[0], axes[0]);
			} else if (node->name == "dim_norm") {
				result = ComputeNorm(inputs[0], axes[0]);
			} else if (node->name == "dim_max") {
				result = ComputeMax(inputs[0], axes[0]);
			} else if (node->name == "dim_min") {
				result = ComputeMin(inputs[0], axes[0]);
			} else if (node->name == "dim_mean") {
				result = ComputeMean(inputs[0], axes[0]);
			} else if (node->name == "dim_product") {
				result = ComputeProduct(inputs[0], axes[0]);
			} else if (node->name == "dim_any") {
				result = ComputeAny(inputs[0], axes[0]);
			} else if (node->name == "dim_all") {
				result = ComputeAll(inputs[0], axes[0]);
			} else if (node->name == "dim_prefix_sum") {
				result = ComputePrefixSum(inputs[0], axes[0]);
			} else if (node->name == "transpose") {
				//get the permutation
				int dim = (int)inputs[0]->GetDimension();
				dim = std::max(dim, std::max(axes[0], axes[1]) + 1);
				map<int, int> permutation;
				for (int i = 0; i < dim; i++) {
					if(i == axes[0]) {
						permutation[i] = axes[1];
					} else if(i == axes[1]) {
						permutation[i] = axes[0];
					} else {
						permutation[i] = i;
					}
				}
				result = Transpose(inputs[0], permutation);
			} else if (node->name == "dot") {
				result = ComputeDot(inputs[0], inputs[1], axes[0]);
			} else if (node->name == "matmul") {
				result = ComputeMatMul(inputs[0], inputs[1]);
			} else if (node->name == "unsqueeze") {
				map<int, int> permutation;
				int dim = (int)inputs[0]->GetDimension()+1;
				dim = std::max(dim, axes[0] + 1);
				for(int i = 0; i < dim; i++) {
					if(i == axes[0]) {
						permutation[i] = 0;
					} else if (i < axes[0]) {
						permutation[i] = i + 1;
					} else {
						permutation[i] = i;
					}
				}
				result = Transpose(inputs[0], permutation);
				result->SetDebugName("unsqueezed");
			} else if (node->name == "squeeze") {
				map<int, int> permutation;
				int dim = (int)inputs[0]->GetDimension() - 1;
				for(int i = 0; i < dim; i++) {
					if(i < axes[0]) {
						permutation[i] = i;
					} else {
						permutation[i] = i + 1;
					}
				}
				result = Transpose(inputs[0], permutation);
				result->SetDebugName("squeezed");
			} else if (node->name == "dim_reverse") {
				result = ReverseDim(inputs[0], axes[0]);
			} else {
				throw std::runtime_error("Unknown algorithmic primitive " + node->name);
			}

			//replace the node with the sum
			node->MakeOutputsUseGivenNode(result->node_);

			ShapeCompareResult shape_result = CompareShape(node, result->node_, true);
			if (!shape_result.compatible) {
				throw std::runtime_error("Algorithmic primitive " + node->name + " at " + node->debug_name + " has incompatible shapes");
			}

			//copy over all memory flags to the new node
			//TODO make a function for this
			result->node_->memory_type_ = node->memory_type_;
			result->node_->special_indices_ = node->special_indices_;

			if (node->debug_name != "") {
				result->node_->debug_name = node->debug_name;
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
}

#define MAX_UNROLL_SIZE 8
#define MAX_UNROLL_NODES 128
void IR::UnrollLoops()
{
	vector<Node*> loops = GetNodesOfType("loop");

	unordered_set<Node*> loops_to_remove;

	for (auto loop : loops) {
		//get inputs (begin, end, step)
		map<int, const Tensor*> inputs = loop->args.GetTensors(ArgType::Input);

		//try get the constant values
		bool is_const = isConstant(inputs[0]) && isConstant(inputs[1]) && isConstant(inputs[2]);

		if (!is_const) {
			continue;
		}

		int begin = inputs[0]->TryGetConstant();
		int end = inputs[1]->TryGetConstant();
		int step = inputs[2]->TryGetConstant();

		//how many iterations to unroll
		int iters = (end - begin) / step;
		if (iters > MAX_UNROLL_SIZE) {
			continue;
		}

		//get all children of the loop
		vector<Node*> children = GetChildren(loop);

		if (children.size() > MAX_UNROLL_NODES) {
			continue;
		}

		//check if they are not keywords or have no children
		set<Node*> nodes_to_copy;
		bool can_unroll = true;
		for (auto child : children) {
			if (child->op->HasAllTypes(OpClass::Keyword) || child->child->valid()) {
				can_unroll = false;
				break;
			}
			nodes_to_copy.insert(child);
		}

		if (!can_unroll) {
			continue;
		}

		//unroll the loop
		ExecuteExpressionAfter(loop, [&]() {
			for (int i = begin; i < end; i += step) {
				unordered_map<Node*, Node*> arg_remap;
				Tensor* index = &Tensor::Constant(i);
				//index->SetDebugName(loop->debug_name + "_unroll_" + to_string(i));
				arg_remap[loop] = index->node_;
				CopyNodes(nodes_to_copy, arg_remap, {}, {}, false);
			}
		});

		//mark the loop for removal
		loops_to_remove.insert(loop);
	}

	// remove all loops that are not used
	for (auto* loop : loops_to_remove) {
		RemoveNode(loop);
	}

	UpdateGraph();
}

void IR::TryReplaceModificationsWithVersions()
{
	UpdateGraph();

	//get all "set" nodes
	vector<Node*> nodes = GetNodesOfType("set");

	unordered_set<Node*> nodes_to_remove;

	for (auto set_node : nodes) {
		//look up the memory node
		Node* memory_node = set_node->args.Get(ArgType::Memory);
		Node* input_value = set_node->args.Get(ArgType::Input);

		//if this node has the same parent as the memory node, then it can be replaced with a version
		if (memory_node->parent == set_node->parent) {
			//replace the set node with the memory node
			ExecuteExpressionBefore(set_node, [&]() {
				Tensor& copied = Tensor::copy(*input_value->GetTensor());
				Node* copynode = copied.node_;
				memory_node->MakeOutputsUseGivenNode(copynode, set_node->index_, true);
				copynode->memory_type_ = memory_node->memory_type_;
				copynode->special_indices_ = memory_node->special_indices_;
				copynode->debug_name = memory_node->debug_name;
				memory_node->memory_type_ = MemoryType::None;
				nodes_to_remove.insert(set_node);
			});
		}	

		UpdateGraph();
	}

	// remove all nodes that are not used
	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();
}

const Tensor& ReduceGradientToShape(const Tensor& gradient, const Tensor& target) {
	ShapeCompareResult shape_result = CompareShape(gradient.node_, target.node_);
	if (!shape_result.compatible) {
		throw std::runtime_error("Autodiff: gradient shape not compatible with target tensor");
	}

	if(!shape_result.broadcast) {
		return gradient;
	}

	int dim = shape_result.broadcast_dim;
	ShapeInfo gradinfo = gradient.GetShapeInfo();
	ShapeInfo targetinfo = target.GetShapeInfo();

	gradinfo.ExpandDimensions(dim);
	targetinfo.ExpandDimensions(dim);

	Tensors grad_shape = gradinfo.GetTensors();
	Tensors target_shape = targetinfo.GetTensors();

	vector<int> axes_to_reduce;
	vector<bool> unsqueeze;
	for(int i = 0; i < dim; i++) {
		int val_a = grad_shape[i]->TryGetConstant();
		int val_b = target_shape[i]->TryGetConstant();
		if(val_a != val_b && val_b == 1) { //if the target has a dimension of 1, and the gradient has a different dimension, then reduce
			axes_to_reduce.push_back(i);
			bool should_unsqueeze = i >= (dim - target.GetDimension());
			unsqueeze.push_back(should_unsqueeze);
		}
	}

	Tensor* reduced = const_cast<Tensor*>(&gradient);
	//go in inverse order to keep the dimensions in the same order
	for(int i = (int)axes_to_reduce.size() - 1; i >= 0; i--) {
		reduced = &Tensor::Sum(*reduced, axes_to_reduce[i]);
		if(unsqueeze[i]) {
			reduced = &Tensor::Unsqueeze(*reduced, axes_to_reduce[i]);
		}
	}

	return *reduced;
}

class NodeGrads
{
	unordered_map<ArgID, Tensor*, HashArgID> argument_gradients;
	unordered_map<ArgID, const Tensor*, HashArgID> argument_inputs;
public:
	//get element at index
	const Tensor& operator[](ArgID id) {
		return *argument_gradients[id];
	}

	bool Contains(ArgID id) {
		return argument_gradients.contains(id);
	}

	bool Contains(ArgType type, int index = 0) {
		return Contains(ArgID(type, index));
	}

	NodeGrads(Node* node, map<Node*, Tensor*> input_grads) {
		for(auto& [id, input] : node->args.inputs_) {
			argument_inputs[id] = input->GetTensor();
			if(input_grads.contains(input)) {
				argument_gradients[id] = input_grads[input];
			}
		}
	}

	void Add(ArgType type, int index, Tensor& tensor) {
		const Tensor* target = argument_inputs[ArgID(type, index)];
		Tensor& new_tensor = const_cast<Tensor&>(ReduceGradientToShape(tensor, *target));
		if(Contains(type, index)) {
			argument_gradients[ArgID(type, index)] = &(*argument_gradients[ArgID(type, index)] + new_tensor);
		} else {
			argument_gradients[ArgID(type, index)] = &new_tensor;
		}
	}

	Tensor* GetGrad(ArgID id) {
		if(Contains(id)) {
			return argument_gradients[id];
		} else {
			Tensor* zero_grad = &Tensor::Constant(argument_inputs[id]->GetShape(), 0.0f);
			argument_gradients[id] = zero_grad;
			return zero_grad;
		}
	}

	Tensor* GetGrad(ArgType type, int index) {
		return GetGrad(ArgID(type, index));
	}

	//add gradients to inputs
	template <typename... Args>
	void Add(Tensor& arg, Args&... args) {
		//by default these are ArgType::Input
		vector<Tensor*> inputs = vector<Tensor*>({ &arg, &args... });
		for (int i = 0; i < inputs.size(); i++) {
			Add(ArgType::Input, i, *inputs[i]);
		}
	}
};

map<string, function<void(ArgumentManager&, Tensor&, Tensor&, NodeGrads&)>> gradient_functions =
{
	//elementwise operations
    {"copy", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad); }},
	{"add", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad, grad); }},
	{"sub", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad, -grad); }},
	{"mul", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * in[1], grad * in[0]); }},
	{"div", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad / in[1], -grad * in[0] / (in[1] * in[1])); }},
	{"neg", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(-grad); }},
	{"exp", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * out); }},
	{"log", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad / in[0]); }},
	{"sin", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * Tensor::cos(in[0])); }},
	{"cos", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(-grad * Tensor::sin(in[0])); }},
	{"tan", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * (Tensor::Constant(1.0f) + out * out)); }},
	{"asin", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad / Tensor::sqrt(Tensor::Constant(1.0f) - out * out)); }},
	{"acos", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(-grad / Tensor::sqrt(Tensor::Constant(1.0f) - out * out)); }},
	{"atan", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad / (Tensor::Constant(1.0f) + out * out)); }},
	{"abs", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * Tensor::sign(in[0])); }},
	{"sign", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f)); }},
	{"exp2", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * Tensor::log(Tensor::Constant(2.0f)) * out); }},
	{"log2", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad / (in[0] * Tensor::log(Tensor::Constant(2.0f)))); }},
	{"sqrt", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad / (Tensor::Constant(2.0f) * out)); }},
	{"rsqrt", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(-grad / (Tensor::Constant(2.0f) * in[0] * out)); }},
	{"floor", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f)); }},
	{"ceil", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f)); }},
	{"round", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f)); }},
	{"frac", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f)); }},
	{"atan2", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * in[1] / (in[0] * in[0] + in[1] * in[1]), -grad * in[0] / (in[0] * in[0] + in[1] * in[1])); }},
	{"lerp", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * in[2], grad * (Tensor::Constant(1.0f) - in[2]), grad * (in[0] - in[1])); }},
	{"max", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::select(in[0] > in[1], grad, Tensor::Constant(0.0f)), Tensor::select(in[0] < in[1], grad, Tensor::Constant(0.0f))); }},
	{"min", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::select(in[0] < in[1], grad, Tensor::Constant(0.0f)), Tensor::select(in[0] > in[1], grad, Tensor::Constant(0.0f))); }},
	{"pow", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * in[1] * Tensor::pow(in[0], in[1] - Tensor::Constant(1.0f)), grad * Tensor::log(in[0]) * out); }},
	{"tanh", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * (Tensor::Constant(1.0f) - out * out)); }},
	{"clamp", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		//clamp = min(max(x, min), max)
		Tensor& dc_dx = Tensor::select((in[0] < in[1]) || (in[0] > in[2]), Tensor::Constant(0.0f), grad);
		Tensor& dc_dmin = Tensor::select(in[0] < in[1], grad, Tensor::Constant(0.0f));
		Tensor& dc_dmax = Tensor::select(in[0] > in[2], grad, Tensor::Constant(0.0f));
		grads.Add(dc_dx, dc_dmin, dc_dmax);
	}},
	{"ternary", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f), Tensor::select(in[0], grad, Tensor::Constant(0.0f)), Tensor::select(in[0], Tensor::Constant(0.0f), grad)); }},
	{"lerp", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * in[2], grad * (Tensor::Constant(1.0f) - in[2]), grad * (in[0] - in[1])); }},
	{"smoothstep", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		//smoothstep equation:
		//t = (x - e0) / (e1 - e0)
		//tc = clamp(t, 0.0, 1.0);
		//r = tc * tc * (3 - 2 * tc);
		//derivative of smoothstep:
		//dr/dx = dr/dtc * dtc/dt * dt/dx
		//dr/dtc = 6 * tc * (tc - 1)
		//dtc/dt = select((t < e0) || (t > e1), 0.0, 1.0)
		//dt/dx = 1 / (e1 - e0)
		//dt/dedge0 = (x - e1) / (e1 - e0)^2
		//dt/dedge1 = (e0 - x) / (e1 - e0)^2
		const Tensor& e0 = in[0];
		const Tensor& e1 = in[1];
		const Tensor& x = in[2];
		const Tensor& t = (x - e0) / (e1 - e0);
		const Tensor& tc = Tensor::clamp(t, Tensor::Constant(0.0f), Tensor::Constant(1.0f));
		const Tensor& dr_dtc = Tensor::Constant(6.0f) * tc * (tc - Tensor::Constant(1.0f));
		const Tensor& dtc_dt = Tensor::select((t < e0) || (t > e1), Tensor::Constant(0.0f), Tensor::Constant(1.0f));
		const Tensor& grad_dt = grad * dr_dtc * dtc_dt;
		const Tensor& dt_dx = Tensor::Constant(1.0f) / (e1 - e0);
		const Tensor& dt_de0 = (x - e1) * (dt_dx * dt_dx);
		const Tensor& dt_de1 = (e0 - x) * (dt_dx * dt_dx);
		grads.Add( grad_dt * dt_de0, grad_dt * dt_de1, grad_dt * dt_dx);
	}},
	{"step", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f), Tensor::Constant(0.0f)); }},
	{"modf", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad, Tensor::Constant(0.0f)); }},
	{"fma", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(in[1] * grad, in[0] * grad, grad); }},

	//matrix operations
	{"matmul", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Matmul(grad, Tensor::Transpose(in[1])), Tensor::Matmul(Tensor::Transpose(in[0]), grad));
	}},
	{"transpose", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Transpose(grad, out.node_->data[1], out.node_->data[0]));
	}},
	{"dot", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		Tensor& unsq_grad = Tensor::Unsqueeze(grad, out.node_->data[0]);
		grads.Add(unsq_grad * in[1], unsq_grad * in[0]);
	}},
	{"unsqueeze", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Sqeeze(grad, out.node_->data[0]));
	}},
	{"dim_sum", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Unsqueeze(grad, out.node_->data[0]));
	}},
	{"dim_mean", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		int axis = (int)out.node_->data[0];
		Tensors shape = in[0].GetShape();
		axis = GetAxis((int)shape.size(), axis);
		Tensor& dim_size = Tensor::tofloat(*shape[axis]);
		grads.Add(Tensor::Unsqueeze(grad / dim_size, axis));
	}},
	{"dim_norm", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		Tensor& unsq = Tensor::Unsqueeze(grad/out, out.node_->data[0]);
		grads.Add(unsq * in[0]);
	}},
	{"dim_max", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Unsqueeze(Tensor::select(in[0] == out, grad, Tensor::Constant(0.0f)), out.node_->data[0]));
	}},
	{"dim_min", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Unsqueeze(Tensor::select(in[0] == out, grad, Tensor::Constant(0.0f)), out.node_->data[0]));
	}},
	{"dim_prefix_sum", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		//b_i = a_0 + ... + a_i
		//db_i/da_j = 1 if i >= j, 0 otherwise
		//dL/da_j = sum_i dL/db_i * db_i/da_j
		//dL/da_j = sum_i dL/db_i * (i >= j)
		//g_i == dL/db_i
		//dL/da_j = g_j + g_{j+1} + ... + g_n = g_n + g_{n-1} + ... + g_j
		//c_i == g_{n-i}
		//dL/da_j = c_0 + c_1 + ... + c_j = prefix_sum(c)_j
		grads.Add(Tensor::PrefixSum(Tensor::Reverse(grad, out.node_->data[0]), out.node_->data[0]));
	}},
	{"dim_reverse", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Reverse(grad, out.node_->data[0]));
	}},
	//memory operations
	{"load", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		//derivative of load is scatter gradient to the load memory addresses
		int index_count = in.Count(ArgType::Index);

		Tensors tensor_indices = Tensors();
		for (int i = 0; i < index_count; i++) {
			tensor_indices.push_back(in.GetTensor(ArgType::Index, i));
		}

		Tensor& curGrad = *grads.GetGrad(ArgType::Memory, 0);
		Tensor::ScatterAdd(curGrad, grad, tensor_indices);
	}},
	{"store", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		//derivative of store is load gradient at the store memory addresses
		const Tensor* memory_input = in.GetTensor(ArgType::Memory);
		int index_count = in.Count(ArgType::Index);

		Tensors tensor_indices = Tensors();
		for (int i = 0; i < index_count; i++) {
			tensor_indices.push_back(in.GetTensor(ArgType::Index, i));
		}

		Tensor& memory_grad = *grads.GetGrad(ArgType::Memory, 0);
		grads.Add(ArgType::Input, 0, Tensor::Load(memory_grad, tensor_indices));
	}},
	{"InterlockedAdd", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		//derivative of scatter_add is load gradient at the scatter memory addresses
		const Tensor* memory_input = in.GetTensor(ArgType::Memory);
		int index_count = in.Count(ArgType::Index);

		Tensors tensor_indices = Tensors();
		for (int i = 0; i < index_count; i++) {
			tensor_indices.push_back(in.GetTensor(ArgType::Index, i));
		}

		Tensor& memory_grad = *grads.GetGrad(ArgType::Memory, 0);
		grads.Add(ArgType::Input, 0, Tensor::Load(memory_grad, tensor_indices));
	}},
	{"set", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		//derivative of set is the gradient of the setted value to the input
		Tensor& memory_grad = *grads.GetGrad(ArgType::Memory, 0);
		grads.Add(ArgType::Input, 0, memory_grad);
	}},
	{"detached_grad", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
	}},
	{"passthrough_grad", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		grads.Add(grad);
	}},
};

void ComputeNodeGradients(Node* value, Tensor* grad, NodeGrads& grads)
{
	string op_name = value->name;
	//add input arguments
	if(value->HasFlags(NodeFlags::PassGrad)) {
		op_name = "passthrough_grad";
	}
	if(value->HasFlags(NodeFlags::DetachGrad)) {
		op_name = "detached_grad";
	}

	if (!gradient_functions.contains(op_name)) {
		throw std::runtime_error("Cannot compute gradient for operation " + op_name);
	}

	Tensor out = *value->tensor_;
	gradient_functions[op_name](value->args, out, *grad, grads);
}

void IR::ComputeAutodiff()
{
	vector<Node*> gradients = GetNodesOfType(OpClass::Gradient);

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
		map<Node*, Tensor*> node_to_grad;

		unordered_set<Node*> loss_deps = GetDependencies({loss});

		//get all differentiable nodes that can change the loss
		vector<Node*> queue;
		for (auto dep : loss_deps) {
			bool in_range = (dep->index_ <= loss->index_ && dep->index_ >= min_range[loss]);
			bool dep_is_accessible = dep->HasCommonParents(loss); //is it in scope of the loss
			if(in_range && !dep->op->HasAllTypes(OpClass::Nondiff) &&
			   dep_is_accessible && (dep->type == TFType::Float || dep->op->HasAllTypes(OpClass::Modifier))) {
				queue.push_back(dep);
			}
		}

		//sort the nodes by index in descending order (backpropagation)
		ranges::sort(queue.begin(), queue.end(), [](Node* a, Node* b) {
			return a->index_ > b->index_;
		});

		Node* loss_value = loss;
		if(loss->op->HasAllTypes(OpClass::Modifier)) {
			loss_value = loss->args.Get(ArgType::Memory);
		}

		ExecuteExpressionAfter(loss, [&]() {
			node_to_grad[loss_value] = &Tensor::Constant(1.0f);
			for(auto node : queue) {
				if(!node_to_grad.contains(node) && !node->op->HasAllTypes(OpClass::Modifier)) {
					continue;
				}

				NodeGrads grads = NodeGrads(node, node_to_grad);
				ComputeNodeGradients(node, node_to_grad[node], grads);

				//store the computed gradients
				for (auto& [id, input]: node->args.inputs_) {
					if(!grads.Contains(id)) {
						continue;
					}

					Tensor& new_grad = *grads.GetGrad(id);
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
		gradient->MakeOutputsUseGivenNode(computed_grad);

		//copy over all memory flags to the new node
		//TODO make a function for this
		computed_grad->memory_type_ = gradient->memory_type_;
		computed_grad->special_indices_ = gradient->special_indices_;

		if (gradient->debug_name != "") {
			computed_grad->debug_name = gradient->debug_name;
		}

		//mark the node for removal
		nodes_to_remove.insert(gradient);

		UpdateGraph();
	}

	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();
}

void IR::CompileIR() 
{
	// TODO (Moroz): Add auto tests into build system

	CheckIR("Input", false, false);
	GetInputList();
	OptimizeOperations();
	//CheckIR("Optimize operations", false, false);
	TryReplaceModificationsWithVersions();
	RemoveUnusedOperations();
	//CheckIR("Remove Unused Operations 0", false, false);
	ComputeAutodiff();
	RemoveUnusedOperations();
	CheckIR("Compute Autodiff", false, false);
	InsertAlgorithmicPrimitives();
	CheckIR("Insert Algorithmic Primitives", false, false);
	UnrollLoops();
	TryReplaceModificationsWithVersions();
	RemoveUnusedOperations();
	CheckIR("Remove Unused Operations 1", false, false);
	SeparateOperationsIntoKernels();
	CheckKernelShapes();
	//UnrollDimensions();
	CheckIR("Separate Operations Into Kernels", false, false);
	ReorderOperations();
	CheckIR("Reorder Operations", true, false);
	MoveShapeOutsideKernels();
	OptimizeKernels(); //fuse kernels by copying inputs
	OptimizeHost();
	CheckIR("Optimize kernels and host", true, false);
	for (int i = 0; i < 10; i++) { //fusing kernels by loads (tensor product)
		RemoveUnusedOperations();
		AddKernelGlobalLoadOperations();
		AddMemoryOpIndices();
		//CheckIR("Load optimization 1 iteration " + to_string(i), true, false);
		OptimizeKernelLoadOperations();
		//CheckIR("Load optimization 2 iteration " + to_string(i), true, false);
	}
	AddKernelGlobalStoreOperations();
	RemoveUnusedKernels();
	CheckIR("Add Kernel Global Memory Operations", true, true);
	AddMemoryOpIndices();
	ReorderOperations();
	OptimizeOperations();
	AddMemoryOpIndices();
	//CheckIR("Final optimization", true, true);
	FinalizeMemoryIndexing();
	RemoveUnusedOperations();
	//CheckIR("Finalize Memory Indexing", false, false);
	OptimizeKernels();
	OptimizeHost();
	//OptimizeLoops();
	RemoveUnusedOperations();
	//CheckIR("Finalize Memory Indexing 2", true, true);
	RemoveUnusedKernels();
	OptimizeOperations();
	RemoveUnusedOperations();
	//CheckIR("Remove Unused Operations 2", true, true);
	AddMemoryDeallocation();
	CheckIR("Add deallocation", true, true);
	GetOutputList();
	ComputeStatistics();
}

Program* GenerateProgram(IR* ir) 
{
	ir->CompileIR();

	auto* program = new Program(ir);

	vector<Node*> kernels = ir->GetNodesOfType("kernel");

	for (auto kernel : kernels)
	{
		// get the kernel type
		map<Node*, size_t> variables;
		map<Node*, bool> read_write;
		NodeArguments shape = kernel->args.GetArguments(ArgType::Shape);
		size_t variable_index = 0;

		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			if (node->op->HasAllTypes(OpClass::MemoryOp)) {
				// get the memory node
				const Tensor* memory = node->args.GetTensor(ArgType::Memory);

				if(node->op->HasAllTypes(OpClass::Modifier)) {
					read_write[memory->node_] |= true;
				} else {
					read_write[memory->node_] |= false;
				}
			}

			// get all input arguments
			for (auto [id, from] : node->args.inputs_) {
				if (id.first == ArgType::Input)
				{
					bool from_outside_kernel = !from->HasParent(kernel);
					if (from_outside_kernel && !variables.contains(from)) {
						variables[from] = variable_index++;
					}
				}
			}
		}

		map<Node*, size_t> read_write_memory;
		map<Node*, size_t> read_only_memory;
		size_t read_write_index = 0;
		size_t read_only_index = 0;
		for(auto [node, rw] : read_write) {
			if(rw) {
				read_write_memory[node] = read_write_index++;
			} else {
				read_only_memory[node] = read_only_index++;
			}
		}

		// add the kernel to the program
		program->AddKernel(kernel, variables, read_write_memory, read_only_memory, shape);
	}

	return program;
}

}  // namespace TensorFrost