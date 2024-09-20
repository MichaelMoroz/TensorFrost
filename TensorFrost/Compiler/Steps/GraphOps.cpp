#include "Compiler/KernelGen.h"
#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {

void KernelScope::CreateKernel() {
	//create kernel node
	Tensor& tensor = Tensor::Kernel(scope_shape.GetTensors());
	Node* kernel_node = tensor.node_;
	Node* old_child = kernel_node->child;
	kernel_node->child = begin;
	kernel_node->next = end->next;
	begin->parent = kernel_node;
	begin->prev = nullptr;
	end->next->prev = kernel_node;
	end->next = old_child;
	old_child->prev = end;
}

void IR::SeparateOperationsIntoKernels() {

	unordered_set<KernelScope*> kernel_scopes;

	ExecuteExpressionFirstChild(root, [&]() {
		kernel_scopes = KernelScope::ComputeScopes(root).first;
	});

	// create kernel nodes for all kernel scopes
	for (auto scope : kernel_scopes) {
		// create kernel before the scope
		ExecuteExpressionBefore(scope->begin, [&]() {
			scope->CreateKernel();
		});
#ifdef _RELWITHDEBINFO
		if (scope->boundary_nodes.size() > max_kernel_memory_dependencies) {
			cout << current_pass << ": Warning: Kernel created with " << scope->boundary_nodes.size() << " boundary nodes" << endl;
		}
#endif
	}

	UpdateGraph();
}


unordered_set<Node*> IR::ComputeKernelDependencies(Node* kernel) {
	unordered_set<Node*> kernel_deps;

	for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
		//go over all inputs
		unordered_set<Node*> node_deps;
		for (auto& [id, from] : node->args.Inputs()) {
			if(id.first == ArgType::Shape) {
				continue;
			}
			bool is_outside = !from->HasParent(kernel);
			if (is_outside) {
				//check if its not scalar
				if(from->args.Count(ArgType::Shape) == 0) {
					continue;
				}
				node_deps.insert(from);
			} else {
				node_deps.insert(from->memory_deps.begin(), from->memory_deps.end());
			}
		}
		//go over all outputs and add this node as a dependency if those outputs are outside the kernel
		//this means the kernel must also write to an output
		for (auto [edge, to] : node->args.Outputs()) {
			if(edge.first.first == ArgType::Shape) {
				continue;
			}
			if (!to->HasParent(kernel)) {
				node_deps.insert(node.get());
			}
		}
		node->memory_deps = node_deps;
		kernel_deps.insert(node_deps.begin(), node_deps.end());
	}

	kernel->memory_deps = kernel_deps;
	return kernel_deps;
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

#ifdef _RELWITHDEBINFO
		auto deps = ComputeKernelDependencies(kernel);
		if(deps.size() > max_kernel_memory_dependencies) {
			cout << current_pass << ": Warning: Kernel " << kernel->debug_index << " has " << deps.size() << " dependencies" << endl;
		}
#endif
	}


	UpdateGraph();
}

bool IR::LimitKernelMemoryDependencies() {
	UpdateGraph();
	vector<Node*> kernels = GetNodesOfType("kernel");

	int created_kernels = 0;

	for (auto kernel : kernels) {
		unordered_set<Node*> kernel_deps = ComputeKernelDependencies(kernel);

		if(kernel_deps.size() <= max_allowed_memory_dependencies) continue;

		//throw std::runtime_error("Kernel " + to_string(kernel->index_) + " has too many memory dependencies (" + to_string(kernel_deps.size()) + " > " + to_string(max_kernel_memory_dependencies) + ")");

		//reclusterize the kernel
		unordered_set<KernelScope*> kernel_scopes;
		ExecuteExpressionFirstChild(kernel, [&]() {
			kernel_scopes = KernelScope::ComputeScopes(kernel).first;
		});


		//move the child nodes outside of the kernel
		vector<Node*> children = kernel->GetChildren();
		Node* first_child = kernel->child;
		Node* last_child = children.back();
		Node* _next = kernel->next;


		kernel->next = first_child;
		first_child->prev = kernel;
		last_child->next = _next;
		_next->prev = last_child;

		kernel->child = nullptr;
		for (auto child : children) {
			child->parent = kernel->parent;
		}

		for (auto scope : kernel_scopes) {
			// create kernel before the scope
			ExecuteExpressionBefore(scope->begin, [&]() {
				scope->CreateKernel();
			});
			created_kernels++;
		}

		UpdateGraph();
	}

	UpdateGraph();

	return created_kernels == 0;
}

void IR::CheckIR(string name, bool check_clustering, bool check_kernels) {
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
		for (auto& [id, input] : node->args.Inputs()) {
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
			for (auto& [id, from] : node->args.Inputs()) {
				bool outside_kernel = !from->HasParent(kernel);
				if (outside_kernel && !node->args.CannotMoveArgument(id)) {
					// if this node is a set and its input is outside of the cluser ->
					// move it inside
					if (node->op->HasAllTypes(OpProp::Set)) {
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
	bool valid = true;
	std::function<void(Node*)> dfs = [&](Node* node) {
		if (nodes_to_copy.contains(node)) return;
		nodes_to_copy.insert(node);
		for (auto& [arg, from] : node->args.Inputs()) {
			if (node->args.CannotCopyArgument(arg)) {
				continue;
			}
			dfs(from);
		}
	};

	for (Node* target : targets) {
		dfs(target);
	}


	return CopyNodes(nodes_to_copy, {}, indices, targets, true);
}

map<Node*, Node*> IR::CopyNodesWithIndex(unordered_set<Node *> nodes_to_copy,
                                         unordered_map<int, Node *> indices,
                                         Node *cursor) {
	// copy all the nodes at the beginning of the kernel
	map<Node*, Node*> copied_node_map;
	if(cursor == nullptr) {
		copied_node_map = CopyComputation(nodes_to_copy, indices);
	} else {
		ExecuteExpressionBefore(cursor, [&]() {
			copied_node_map = CopyComputation(nodes_to_copy, indices);
		});
	}
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

	ReplaceArgs(args_to_copy, copied_node_map);
}

void IR::MoveShapeOutsideKernels() {
	UpdateGraph();
	// find all nodes that are used as shapes and are inside kernels
	map<Node*, Node*> nodes_to_copy;
	for (auto node = begin(); !node.end(); node.next()) {
		Node* kernel = node->GetParent("kernel");
		if (kernel == *node) { //if returns itself, then no kernel parent found
			continue;
		}

		// go over all outputs arguments
		for (auto [edge, to] : node->args.Outputs()) {
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
		for (auto [edge, to] : node->args.Outputs()) {
			auto& [id, from] = edge;
			if (id.first == ArgType::Shape) {
				args_to_copy.insert(ArgEdge(Arg(id, node), to));

				//get the earliest output
				if (to->index_ < earliest_output_index) { //wat
					earliest_output_index = to->index_;
					earliest_output = to;
				}
			}
		}

		Node* common_parent = earliest_output->GetNodeWithCommonParent(kernel);

		// copy shape computation and put it before the earliest output (outside of the kernel if its inside)
		CopyArguments(args_to_copy, common_parent);
		ApplyChanges(false);
	}
}


/// <summary>
/// Get all inputs of this program in the IR
/// </summary>
void IR::GetInputList() {
	int input_memory_index = 0;
	//MUST BE IN ORDER
	for (auto node = begin(); !node.end(); node.next()) {
		if (node->flags.has(NodeProp::InputMemory)) {
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
			node->flags.set(NodeProp::InputMemory, input_index);
			//if any of the inputs are "input_shape" then we need to add the input index to them
			for (auto& [arg, from] : node->args.Inputs()) {
				if (arg.first == ArgType::Shape && from->name == "input_shape") {
					if(!from->flags.has(NodeProp::InputShapeMemory)) { //ONLY FIRST TIME
						from->flags.set(NodeProp::InputShapeMemory, input_index);
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
		if (node->flags.has(NodeProp::OutputMemory)) {
			if (!node->op->HasAllTypes(OpProp::Memory)) {
				throw std::runtime_error(
				    "Compilation error: output is not a memory node");  // all outputs
				                                                        // should be
				                                                        // memory nodes
				                                                        // at this point
			}
			output_memory_map[node->flags.get(NodeProp::OutputMemory)] = *node;
		}
		if (node->op->HasAllTypes(OpProp::Modifier, OpProp::MemoryOp)) {
			if (!node->HasParent("kernel")) {
				writebacks++;
			}
		} else if (node->op->HasAllTypes(OpProp::Load, OpProp::MemoryOp)) {
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
			bool is_input = node->flags.has(NodeProp::InputMemory);
			bool is_output = node->flags.has(NodeProp::OutputMemory);
			if (is_input) {
				input_memory_count++;
			}
			if (!is_input && !is_output) {
				temp_memory_count++;
			}
		}
	}

	//Check if output memory map has all the outputs
	if (output_memory_map.size() < output_memory_count) {
		throw std::runtime_error("Output memory map does not have all the outputs, some got lost");
	} else if (output_memory_map.size() > output_memory_count) {
		throw std::runtime_error("Output memory map has more outputs than expected");
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
		for (auto& [arg, from] : node->args.Inputs()) {
			dfs(from);
		}

		//if the node is a memory node or used as memory, then all outputs are used
		for (auto [edge, to] : node->args.Outputs()) {
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

void IR::ComputeNodeCost()
{
	for (auto node = begin(); !node.end(); node.next()) {
		bool is_memory = node->op->HasAllTypes(OpProp::Memory);
		unordered_map<Node*, float> input_costs;
		for (auto& [id, from] : node->args.Inputs()) {
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

		//go over outputs and check if it has any load operations
		bool is_used_as_memory = false;
		for (auto [edge, to] : node->args.Outputs()) {
			auto& [id, from] = edge;
			if(to->op->HasAllTypes(OpProp::MemoryOp)) {
				is_used_as_memory = true;
				break;
			}
		}

		if(is_used_as_memory) {
			node->flags.set(NodeProp::NoCopyFusion);
		}
	}
}

map<Node *, ArgEdges> IR::GetKernelOutputs(Node *kernel)
{
	map<Node*, ArgEdges> node_output;
	for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
		bool is_output = node->flags.has(NodeProp::OutputMemory);
		ArgEdges outputs = ArgEdges();

		for (auto [edge, to] : node->args.Outputs()) {
			auto& [id, from] = edge;
			if (to == nullptr) continue;
			// if is a shape or memory argument, then skip (shape is loaded on CPU)
			if (id.first == ArgType::Shape) continue;
			if (!to->HasParent(kernel)) {
				outputs.emplace(Arg(id, *node), to);
				is_output = true;
			}
		}

		if (is_output) {
			node_output[*node] = outputs;
		}
	}

	return node_output;
}

string IR::PrintListing(map<Node*, string> node_debug) const {
	return GetOperationListing(*this, false, node_debug) + "\n\n";
}

string IR::GetNodeListing(Node* node) const {
	return GetNodeString(node, true);
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

	//TODO: figure out a better heuristic for this, or add as a warning
	//(need to implement a logging system)
	// if (nodes_to_copy.size() > 1024) {
	// 	throw std::runtime_error(
	// 	    "Copy Nodes: Copying too many nodes, something is probably "
	// 	    "wrong. Number of nodes to copy: " +
	// 	    to_string(nodes_to_copy.size()));
	// }

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
				Node* new_index = indices.at(dim);
				if(new_index == nullptr) {
					throw std::runtime_error("Copy Nodes: New index is null for node " + node->name);
				}
				new_node = new_index;
				no_index = false;
			}
		}

		if (no_index) {
			// create new arguments
			NodeArguments new_args;
			for (auto& [arg, from]: node->args.Inputs()) {
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


void IR::AddNodeLoadOperations(Node* node, Node* kernel, Tensors indices) {
	for (auto& [arg, input_node] : node->args.Inputs()) {
		if (arg.first == ArgType::Memory || arg.first == ArgType::Shape)
			continue;

		bool is_in_a_kernel = input_node->HasParent("kernel");
		bool is_outside = !input_node->HasParent(kernel);
		bool is_memory = input_node->op->HasAllTypes(OpProp::Memory);

		if (is_memory || (is_in_a_kernel && is_outside)) {
			// load the memory node before this node
			ExecuteExpressionBefore(node, [&]() {
				Tensor& loaded = Tensor::Load(*input_node->GetTensor(), indices, IndexingMode::Unsafe);
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
			for (auto& [arg, input_node] : node->args.Inputs()) {
				if (arg.first == ArgType::Memory || arg.first == ArgType::Shape)
					continue;

				bool is_in_a_kernel = input_node->HasParent("kernel");
				bool is_outside = !input_node->HasParent(kernel);
				bool is_memory = input_node->op->HasAllTypes(OpProp::Memory);

				if (is_memory || (is_in_a_kernel && is_outside)) {
					nodes_to_load.insert(input_node);
					load_arguments[input_node].insert(ArgEdge(Arg(arg, input_node), node.get()));
				}
			}
		}

		for (auto node : nodes_to_load) {
			// load the memory node at the beginning of the kernel
			ExecuteExpressionFirstChild(kernel, [&]() {
				Tensor& loaded = Tensor::Load(*node->GetTensor(), {}, IndexingMode::Unsafe);
				for (auto [in, out] : load_arguments[node]) {
					auto& [arg, from] = in;
					out->args.UpdateArgument(arg, loaded.node_);
				}
			});
		}
	}

	UpdateGraph();
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
			if (!node->op->HasAllTypes(OpProp::MemoryOp)) {
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
						index = indices[i];
					}
					node->args.AddArgument(ArgType::Index, i, index->node_);
				}
			});
		}
	}

	UpdateGraph();
}

void IR::AddKernelGlobalStoreOperations() {
	UpdateGraph();
	// get kernels
	vector<Node*> kernels = GetNodesOfType("kernel");

	// go over all outputs of each kernel and create memory nodes to store the
	// output
	for (auto kernel: kernels) {
		map<Node*, ArgEdges> node_output = GetKernelOutputs(kernel);

		for (auto [output, args] : node_output) {
			// if the output is already a memory node, then skip
			if (output->op->HasAllTypes(OpProp::Memory)) {
				continue;
			}

			Node* mem;
			// add memory node before this kernel
			ExecuteExpressionBefore(kernel, [&]() {
				mem = Tensor::Memory(kernel->args.GetArguments(ArgType::Shape), output->type).node_;
				mem->debug_name = output->debug_name;

				if (output->flags.has(NodeProp::OutputMemory)) {
					mem->flags.copy_all_given(output->flags, { NodeProp::OutputMemory });
					output->flags.remove(NodeProp::OutputMemory);
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
						Tensor& loaded = Tensor::Load(*mem->GetTensor(), {}, IndexingMode::Unsafe);
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
			Node* last_mod_parent = last_mod->GetNodeWithCommonParent(output);

			// add store node after the last modification on the same level as the memory
			ExecuteExpressionAfter(last_mod_parent, [&]() {
				// add store node after this node
				Tensor* store = &Tensor::Store(*mem->GetTensor(), *output->GetTensor(), {}, true);
			});
		}
	}

	// replace all inputs pointing to memory nodes with the memory node
	for (auto node = begin(); !node.end(); node.next()) {
		bool is_memory = node->op->HasAllTypes(OpProp::Memory);

		for (auto& [id, from] : node->args.Inputs()) {
			if (id.first == ArgType::Memory ||
			    (id.first  == ArgType::Shape && !is_memory))
				continue;

			if (from->op->HasAllTypes(OpProp::Memory)) {
				// load the memory node before this node
				ExecuteExpressionBefore(node.get(), [&]() {
					Tensor& loaded = Tensor::Load(*from->GetTensor(), {}, IndexingMode::Unsafe);
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
		if (memory->flags.has(NodeProp::InputMemory)) {
			continue;
		}

		Node* last_output = nullptr;
		int last_output_index = -1;

		bool is_an_output = false;

		//do a dfs to find the last output
		std::function<void(Node*)> dfs = [&](Node* node) {
			if (node->flags.has(NodeProp::OutputMemory)) {
				is_an_output = true;
				return;
			}

			for (auto [edge, to] : node->args.Outputs()) {
				auto& [id, from] = edge;
				if (to->op->HasAllTypes(OpProp::MemoryReuse)) {
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
		Node* deallocation_point = last_output->GetNodeWithCommonParent(memory);

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
	sizes[0] = kernel_shape[0];
	for (size_t i = 1; i < dims - 1; i++) {
		sizes[i] = &(*sizes[i - 1] * *kernel_shape[i]);
	}

	Tensor* temp;
	for (size_t i = 0; i < dims; i++) {
		Tensor* idx0 = index;
		if (i < dims - 1) {
			idx0 = &(*idx0 / *sizes[dims - i - 2]);
		}
		if (i > 0) {
			temp = &(*temp * *kernel_shape[dims - i - 1]);
			idx0 = &(*idx0 - *temp);
			if (i != dims - 1) temp = &(*temp + *idx0);
		} else {
			temp = idx0;
		}
		indices[dims - i - 1] = idx0;
	}

	return indices;
}


// compute the flat index (in C-order)
Tensor* ComputeFlatIndex(NodeArguments memory_shape, vector<Tensor*> indices, map<int, const Tensor*> idx, int memory_dim, IndexingMode mode = IndexingMode::Clamp)
{
	if (memory_dim == 0)
	{
		return &Tensor::Constant(0);
	}

	int kernel_dim = (int)indices.size();

	function<const Tensor*(int)> get_shape = [&](int dim) {
		dim = memory_dim - dim - 1;
		return memory_shape[ArgID(ArgType::Shape, dim)]->GetTensor();
	};

	// function to get index for given dimension, if not found then return
	// default dim index
	function<Tensor*(int)> get_index = [&](int dim) {
		int idxdim = memory_dim - dim - 1;
		Tensor* out;
		if (idx.find(idxdim) != idx.end()) {
			out = const_cast<Tensor*>(idx[idxdim]);
		} else {
			throw std::runtime_error("Finalize memory indexing: node index not found for dimension " + to_string(idxdim) + " in memory node with dimensions " + to_string(memory_dim));
		}

		switch (mode)
		{
			case IndexingMode::Clamp:
				return &Tensor::clamp(
				    *out, TensorFrost::Tensor::Constant(0),
				    *get_shape(dim) - TensorFrost::Tensor::Constant(1));
			case IndexingMode::Unsafe:
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
		if (node->name == "dim_id") { // remove the dim node
			nodes_to_remove.insert(node.get());
		}
		else
		{
			//go over node inputs and replace dim nodes with index nodes
			for (auto& [id, from] : node->args.Inputs()) {
				if (from->name == "dim_id") {
					int dim = from->data[0];
					Node* index_node = nullptr;
					if (dim >= dims) { //if dim node of dimension greater than the number of dimensions its 0
						ExecuteExpressionBefore(node.get(), [&]() {
							index_node = Tensor::Constant(0).node_;
						});
					} else {
						index_node = indices[dim]->node_;
					}

					// replace the dim node with the index node
					node->args.UpdateArgument(id, index_node);
				}
			}
		}
	}

	// remove all dim nodes
	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}
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
	}
	vector<Tensor*> in_block_indices;
	for (int i = 0; i < block_dim; i++) {
		in_block_indices.push_back(&block_index->BlockThreadIndex(i));
	}

	//compute out-of-block index
	Tensors blocks_shape = {};
	for (int i = 0; i < block_dim; i++) {
		const Tensor block_size = *block_size_tensors[i];
		const Tensor shape = *kernel_shape[i];
		Tensor& ceil = (shape + block_size - Tensor::Constant(1)) / block_size;
		blocks_shape.push_back(&ceil);
		blocks_shape[i]->SetDebugName("blocks_shape_" + to_string(i));
	}
	for (int i = block_dim; i < dims; i++) {
		blocks_shape.push_back(kernel_shape[i]);
		blocks_shape[i]->SetDebugName("blocks_shape_" + to_string(i));
	}
	vector<Tensor*> out_block_indices = ComputeIndicesFromLinearIndex(block_index, blocks_shape, dims);

	//combine the final indices
	vector<Tensor*> indices = {};

	for (int i = 0; i < block_dim; i++) {
		indices.push_back(&(*out_block_indices[i] * *block_size_tensors[i] + *in_block_indices[i]));
		indices[i]->SetDebugName("index_" + to_string(i));
	}

	for (int i = block_dim; i < dims; i++) {
		indices.push_back(out_block_indices[i]);
		indices[i]->SetDebugName("index_" + to_string(i));
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
			int shape = kernel_shape[i]->TryGetConstant();
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

void IR::ComputeAddress(Node* node, vector<Tensor*> indices)
{
	// get the input memory node
	const Tensor* memory = node->args.GetTensor(ArgType::Memory);

	NodeArguments memory_shape = memory->node_->args.GetArguments(ArgType::Shape);

	int memory_dim = (int)memory_shape.size();

	// get the index nodes
	map<int, const Tensor*> idx = node->args.GetTensors(ArgType::Index);

	if (idx.empty())
	{
		node->indexing_mode_ = IndexingMode::Unsafe; //we can guarantee that the index is in bounds
	}

	auto flat_index= ComputeFlatIndex(memory_shape, indices, idx, memory_dim, node->indexing_mode_);

	// TODO(Moroz): add wrap mode

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
			if (node->op->HasAllTypes(OpProp::MemoryOp)) {
				ExecuteExpressionBefore(*node, [&]() { ComputeAddress(node.get(), indices); });
			}
		}
	}

	//now compute address for all nodes that are not in a kernel
	for (auto node = begin(); !node.end(); node.next()) {
		if (!node->HasParent("kernel") && node->op->HasAllTypes(OpProp::MemoryOp)) {
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
				memory_node->ReplaceThisWithGivenNode(copynode, set_node->index_, true);
				nodes_to_remove.insert(set_node);
			});
		}

		UpdateIndex();
	}

	UpdateGraph();

	// remove all nodes that are not used
	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();
}

void IR::ApplyChanges(bool update_graph, const Node* uroot) {
	for (auto& [arg, out] : edgesToUpdate) {
		Node* from = arg.second;
		if (!replacementNodes.contains(from)) {
			throw std::runtime_error("No replacement node found for node " + from->name);
		}
		Node* to = replacementNodes[from];
		out->args.UpdateArgument(arg.first, to);
	}

	// remove all nodes that are not used
	for (auto* node : removedNodes) {
		RemoveNode(node);
	}

	if (update_graph) {
		UpdateGraph(uroot);
	}

	ClearChanges();
}

void IR::ClearChanges() {
	edgesToUpdate.clear();
	removedNodes.clear();
	replacementNodes.clear();
}

} // namespace TensorFrost