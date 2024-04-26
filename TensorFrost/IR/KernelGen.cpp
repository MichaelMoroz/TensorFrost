#pragma once

#include "IR/KernelGen.h"

namespace TensorFrost {

[[nodiscard]] const Tensor* Node::GetTensor() const {
	if (tensor_->node_ != this) {
		throw std::runtime_error("Fatal Error: Tensor node does not match");
	}
	return tensor_;
}

ShapeCompareResult CompareShape(const Node* a, const Node* b, bool exact_match, bool throw_error) {
	ArgMap a_shape = a->GetArgumentMap(ArgType::Shape);
	ArgMap b_shape = b->GetArgumentMap(ArgType::Shape);
	int a_dim = MaxIndexCount(a_shape);
	int b_dim = MaxIndexCount(b_shape);

	ShapeCompareResult result;
	result.compatible = true;
	result.is_broadcast = false;
	result.a_dim = a_dim;
	result.b_dim = b_dim;


	int min_dim = min(a_dim, b_dim);

	if (exact_match && min_dim > 0) {
		if (a_dim != b_dim) {
			result.compatible = false;
			if (throw_error) {
				throw std::runtime_error("Shapes must have the same dimension for " + a->var_name + " and " + b->var_name);
			}
			return result;
		}
	}

	for (int i = 0; i < min_dim; i++) {
		Node* a_node = a_shape[a_dim - i - 1]->from_->get();
		Node* b_node = b_shape[b_dim - i - 1]->from_->get();
		int val_a = -1;
		int val_b = -1;
		if (a_node->name == "const") val_a = a_node->GetTensor()->data[0];
		if (b_node->name == "const") val_b = b_node->GetTensor()->data[0];

		//if one of the nodes is a constant = 1, then it is a broadcast
		if ((val_a == 1 || val_b == 1) && !exact_match) {
			result.is_broadcast = true; 
			continue;
		} 

		// if a and b are constants, then compare their values
		if(val_a != -1 && val_b != -1)
		{
			if (val_a != val_b) {
				result.compatible = false;
				if (throw_error) {
					throw std::runtime_error(
					    "Constant dimensions are not compatible for nodes: " +
					    a->var_name + " and " + b->var_name + " at index " +
					    to_string(i) + " with values " + to_string(val_a) + " and " +
					    to_string(val_b));
				}
				return result;
			}
			else
			{
				continue;
			}
		}

		// otherwise, if a and b are not the same node then they are not the same shape (possibly)
		if (a_node != b_node) {
			if (throw_error) {
				throw std::runtime_error("Shapes are potentially not compatible for nodes: " + a->var_name + " and " + b->var_name + " at index " + to_string(i));
			}
			result.compatible = false;
			return result;
		}
	}

	return result;
}

// returns true if the edge between given nodes is a boundary between kernels
bool IsBoundary(Node* input, Node* output,
                ScopeType input_scope = ScopeType::None, bool is_identity = true,
                int arg_index = -1, ArgType arg_type = ArgType::None) {
	ShapeCompareResult result = CompareShape(input, output);

	if (!result.compatible) {
		return true;
	}

	if (input_scope != output->scope_type_ && output->scope_type_ == ScopeType::Kernel) {
		return true;
	}

	// memory should not be inside work kernels
	bool is_output_host_only = output->op->HasAllTypes(OpType::HostOnly);
	bool is_input_host_only = input->op->HasAllTypes(OpType::HostOnly);
	bool is_output_scalar = result.b_dim == 0;

	bool scalar_kernel = output->TryComputeShape() == 1;

	switch (input_scope) { 
		case ScopeType::None:
			return false;
		case ScopeType::Host:
			if (!is_output_scalar && !is_output_host_only) {
				return true; // host should not have non-scalar operations
			}
			break;
		case ScopeType::Kernel:
			if (is_output_host_only || is_input_host_only) {
				return true; // kernel should not have host operations
			}

			const Operation* input_op = input->op;
			const Operation* output_op = output->op;

			if (!scalar_kernel) {
				if (output_op->HasAllTypes(OpType::Load, OpType::MemoryOp)) {
					return arg_type == ArgType::Memory && !is_identity;
				}

				if (output_op->HasAnyType(OpType::Scatter, OpType::Store) &&
				    !input_op->HasAnyType(OpType::Scatter, OpType::Store)) {
					return arg_type == ArgType::Memory;
				}

				// if input has changed the memory and the output is a load then it is a
				// boundary
				if (input_op->HasAllTypes(OpType::MemoryOp, OpType::Modifier) &&
				    output_op->HasAllTypes(OpType::Load, OpType::MemoryOp)) {
					return true;
				}
			}

			if (arg_type == ArgType::Shape) {
				return true;  // shape should not be inside kernels
			}

			break;
	}

	return false;
}

//TODO: rewrite this function
void IR::SeparateOperationsIntoKernels() {
	UpdateGraph();
	vector<Scope*> kernels;
	Scope* current_scope = new Scope(root->child);

	map<Node*, string> boundary_nodes_debug;

	for (auto it = begin(); !it.end(); it.next()) {
		if (it->HasParent("kernel")) {
			continue;
		}

		Node* node = it.get();
		int current_depth = node->ComputeDepth();
		int begin_depth = current_scope->begin->ComputeDepth();
		Arguments indices = node->GetArguments(ArgType::Index);
		bool ident = indices.empty();
		bool loop_prev_iteration = false;

		map<int, Node*> boundary_nodes;

		Node* prev = node->prev;
		if (prev != nullptr) {
			if (current_scope->InScope(prev) &&
			    IsBoundary(prev, node, current_scope->type, ident)) {
				boundary_nodes[prev->index_] = prev;
			}
		}

		ShapeCompareResult result = CompareShape(current_scope->shape_node, node, true);
		if (!result.compatible) {
			boundary_nodes[current_scope->shape_node->index_] =
			    current_scope->shape_node;
		}

		// go over all inputs
		for (auto& input : node->inputs_) {
			// get latest input version
			Node* latest = input.from_->get()->GetLastVersion(node);
			// check if input is the boundary of this kernel
			bool is_loop_boundary =
			    latest->index_ > node->index_ && begin_depth < current_depth;
			if ((current_scope->InScope(latest) || is_loop_boundary) &&
			    IsBoundary(latest, node, current_scope->type, ident, input.index_, input.type_)) {
				if (is_loop_boundary) {
					latest = latest->GetParent("loop");
					if (!current_scope->InScope(latest)) 
					{
						continue;
					}
					loop_prev_iteration = true;
				}
				boundary_nodes[latest->index_] = latest;
			}
		}
		
		if (current_scope->type == ScopeType::Host) {
			current_scope = new Scope(node);
			continue;
		}

		// if boundary, create new scope, else make this new end
		if (boundary_nodes.size() > 0) {
			string boundary_nodes_str = "Boundaries of this node: ";
			int i = 0;
			for (auto& [index, node] : boundary_nodes) {
				if (i > 0) boundary_nodes_str += ", ";
				boundary_nodes_str += node->var_name;
				i++;
			}
			boundary_nodes_debug[node] = boundary_nodes_str;
			
			if (begin_depth > current_depth) {
				Node* last_child = current_scope->begin->parent->GetLastChild();
				NodeIterator it(last_child, last_child);
				kernels.push_back(new Scope(current_scope->begin, it.get()));
				current_scope = new Scope(it.next().get(), node);
			} else if (current_scope->type == ScopeType::Kernel) {
				// split the current scope using the last boundary node (highest index)
				Node* boundary_node = boundary_nodes.rbegin()->second;
				//find the nearest parent node of the scope end with the same parent as the boundary node
				Node* parent = node->GetCommonParent(boundary_node);
				int boundary_depth = boundary_node->ComputeDepth();

				if (boundary_depth > current_depth) {
					parent = parent->next;
				}

				vector<Scope*> new_scopes =
				    Scope::GetScopes(current_scope->begin, parent);
				for (auto scope : new_scopes) {
					if (scope->type == ScopeType::Kernel) {
						kernels.push_back(scope);
					}
				}

				if (loop_prev_iteration)
				{
					current_scope = new Scope(parent->true_next, node);
				}
				else
				{
					current_scope = new Scope(parent, node);
				}
				
			} else { 
				// the current scope was a host scope, can just ignore it, we wont be using it
				current_scope = new Scope(node);
			}
		} else {
			current_scope->UpdateEnd(node);
		}
	}


	if (current_scope->type == ScopeType::Kernel) {
		kernels.push_back(current_scope);
	}

#ifndef NDEBUG
	string listing = PrintListing(boundary_nodes_debug);

	cout << "Kernel genration boundaries: \n\n";
	cout << listing << endl;
#endif

	// create kernel nodes for all kernel scopes
	for (auto scope : kernels) {
		// create kernel node before the scope
		ExecuteExpressionBefore(scope->begin, [&]() {
			//create kernel node
		 	Tensor& tensor = Tensor::Kernel(scope->shape_node->GetArguments(ArgType::Shape));
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
#ifdef NDEBUG
	return "";
#endif
	return GetOperationListing(*this, false, node_debug) + "\n\n";
}

bool BoundaryValid(const Node* input, const Node* output,
                   bool is_identity = true, int arg_index = -1,
                   ArgType arg_type = ArgType::None) {
	//bool same_kernel = input->kernel_ == output->kernel_;
	//bool is_boundary = IsBoundary(input, output, is_identity, arg_index, arg_type);
	//if (output->op->HasAllTypes(OpType::Set) && !same_kernel) return false; // set is always within the same kernel
	//if (!same_kernel) return true;
	//return !is_boundary;
	return true;
}


// check if all child nodes in a kernel have compatible shape to the kernel
void IR::CheckKernelShapes() {
	// get kernels
	UpdateGraph();
	vector<Node*> kernels = GetNodesOfType("kernel");

	// go over all outputs of each kernel and create memory nodes to store the
	// output
	for (auto kernel : kernels) {
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			// check if the node has a shape argument
			ShapeCompareResult result = CompareShape(kernel, node.get());
			if (!result.compatible) {
				throw std::runtime_error("Kernel " + kernel->var_name +
				                         " has incompatible shape with node " +
				                         node.get()->var_name);
			}
		}
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
		Arguments indices = node->GetArguments(ArgType::Index);
		bool identity = indices.empty();

		Node* prev = node->prev;

		if (prev == nullptr) continue;

		if (check_clustering) {
			if (!BoundaryValid(prev, *node, identity)) {
				invalid_nodes[node.get()] = "Invalid node order";
			}
		}

		// go over all inputs
		for (auto& input : node->inputs_) {
			Node* from = input.from_->get();
			Node* to = node.get();

			//if (check_clustering)
			//{
			//	// get latest input version
			//	const Node* latest = input.from_->get()->GetLastVersion(*node);
			//	// check if input is the boundary of this kernel
			//	if (!BoundaryValid(latest, to, identity, input.index_, input.type_)) {
			//		invalid_nodes[to] = "Invalid clusterization for argument " + Arg::TypeToString(input.type_) + ":" + to_string(input.index_);
			//	}
			//}

			//if (check_kernels)
			//{
			//	//check if no inputs are outside the kernel
			//	if (from->kernel_ != to->kernel_ && 
			//		input.type_ != ArgType::Memory && 
			//		input.type_ != ArgType::Shape && from->name != "memory" &&
			//	    from->name != "const") {
			//		invalid_nodes[to] = "Argument " + Arg::TypeToString(input.type_) + ":" + to_string(input.index_) + " is outside the kernel";
			//	}
			//}

			// check if inputs are before the node
			if (from->index_ >= to->index_) {
				if (input.type_ != ArgType::Shape) {
					invalid_nodes[to] = "Argument " + Arg::TypeToString(input.type_) + ":" +
										to_string(input.index_) + " is after the node";
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

bool CannotMoveArgument(Arg& arg) {
	Node* from = arg.from_->get();
	Node* to = arg.to_->get();
	return (arg.type_ == ArgType::Memory &&
	        !to->op->HasAllTypes(OpType::Set)) ||
	       (arg.type_ == ArgType::Shape && !to->op->HasAllTypes(OpType::Memory)) ||
	       from->op->HasAllTypes(OpType::Memory) ||
	       (from->name == "const" && to->op->HasAllTypes(OpType::Memory)); //FIX THIS
}

void IR::ReorderOperations() {
	// get kernel data
	UpdateGraph();
	vector<Node*> kernels = GetNodesOfType("kernel");
	
	for (auto* kernel: kernels) {
		unordered_set<Node*> nodes_to_move;
		// go over all nodes in the kernel and check if their inputs can be copied
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			// go over all inputs
			for (auto& input : node->inputs_) {
				bool outside_kernel = !input.from_->get()->HasParent(kernel);
				if (outside_kernel && !CannotMoveArgument(input)) {
					// if this node is a set and its input is outside of the cluser ->
					// move it inside
					if (node->op->HasAllTypes(OpType::Set)) {
						nodes_to_move.insert(input.from_->get());
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
}

bool CannotCopyArgument(Arg& arg) {
	Node* from = arg.from_->get();
	Node* to = arg.to_->get();
	bool shape = arg.type_ == ArgType::Shape;
	bool to_memory = to->op->HasAllTypes(OpType::Memory);
	bool shape_not_memory = shape && !to_memory;
	return arg.type_ == ArgType::Memory || shape_not_memory || from->op->HasAllTypes(OpType::Static) ||
	       from->op->HasAllTypes(OpType::Memory) || from->HasBeenModified();
}

map<Node*, Node*> IR::CopyComputation(
    const unordered_set<Node*>& targets, const unordered_map<int, Node*>& indices) {
	//if we have indices, we are basically rerunning the computation with a different set of indices (of possible different size)
	bool can_change_shape = !indices.empty();
	Arguments shape_args = Arguments();
	if (can_change_shape) {
		//get first index
		int first_index = indices.begin()->first;
		Node* first_index_node = indices.at(first_index);
		for (auto& arg : first_index_node->inputs_) {
			if (arg.type_ == ArgType::Shape) {
				shape_args.push_back(arg);
			}
		}
	}

	// do a depth first search to copy all the nodes required for the targets
	// (only if in the same kernel)
	unordered_set<Node*> nodes_to_copy;
	std::function<void(Node*)> dfs = [&](Node* node) {
		if (nodes_to_copy.contains(node)) return;
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

	if (nodes_to_copy.size() > 1024) {
		throw std::runtime_error(
		    "Copy Computation: Copying too many nodes, something is probably wrong. Number of nodes to copy: " + to_string(nodes_to_copy.size()));
	}

	// copy the nodes
	map<Node*, Node*> copied_node_map;
	for (auto node = begin(); !node.end(); node.next()) {
		if (!nodes_to_copy.contains(node.get())) {
			continue;
		}

		Node* new_node;
		

		//if we have the index, use it instead
		bool is_dim = node->name == "dim_id";
		bool no_index = true;
		if (is_dim)
		{
			int dim = node->GetTensor()->data[0];
			if (indices.contains(dim))
			{
				new_node = indices.at(dim);
				no_index = false;
			}
		}

		if (no_index) {
			// create new arguments
			Arguments new_args;
			for (Arg& arg : node->inputs_) {
				if (can_change_shape && arg.type_ == ArgType::Shape) {
					continue;
				}

				// if shape or memory argument, then no need to use copied node
				if (CannotCopyArgument(arg) && !targets.contains(arg.from_->get())) {
					new_args.push_back(arg);
					continue;
				}

				Node* from = arg.from_->get();

				if (!copied_node_map.contains(from)) {
					throw std::runtime_error("Copy Computation: Node not found");
				}

				// create new argument
				new_args.emplace_back(arg.type_, copied_node_map[from]->GetLable(),
				                      arg.index_);
			}

			if (can_change_shape) {
				//add shape arguments
				new_args.insert(new_args.end(), shape_args.begin(), shape_args.end());
			}

			// create new node
			Tensor* tensor = Tensor::GetCopy(*node->GetTensor(), new_args);
			new_node = tensor->node_;
		}

		copied_node_map[node.get()] = new_node;
	}

	return copied_node_map;
}

void IR::GetInputList() {
	int input_memory_index = 0;
	for (auto node = begin(); !node.end(); node.next()) {
		if (node->memory_type_ == MemoryType::Input) {
			//add shapes to the memory inputs
			memory_inputs.push_back(*node);

			shape_memory_map[*node] = {};
			//add shapes to the memory inputs
			for (auto& arg : node->inputs_) {
				if (arg.type_ == ArgType::Shape) {
					shape_memory_map[*node][arg.index_] = arg.from_->get();
				}
			}

			//set input memory index
			node->special_index_ = input_memory_index++;
		}
	}
}

void IR::GetOutputList() {
	for (auto node = begin(); !node.end(); node.next()) {
		if (node->memory_type_ == MemoryType::Output) {
			if (!node->op->HasAllTypes(OpType::Memory)) {
				throw std::runtime_error("Compilation error: output is not a memory node"); //all outputs should be memory nodes at this point
			}
			output_memory_map[node->special_index_] = *node;
		}
		if (node->op->HasAllTypes(OpType::Modifier, OpType::MemoryOp)) {
			if (!node->HasParent("kernel")) {
				writebacks++;
			}
		}
		else if (node->op->HasAllTypes(OpType::Load, OpType::MemoryOp)) {
			if (!node->HasParent("kernel")) {
				readbacks++;
			}
		}
	}
}

void IR::ComputeStatistics() {
	for (auto node = begin(); !node.end(); node.next()) {
		if (node->name == "memory")
		{
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

void IR::CopyArguments(unordered_set<Arg*> args_to_copy, Node* cursor)
{
	unordered_set<Node*> nodes_to_copy;
	for (auto* arg : args_to_copy) {
		nodes_to_copy.insert(arg->from_->get());
	}

	// copy all the nodes at the beginning of the kernel
	map<Node*, Node*> copied_node_map;
	unordered_map<int, Node*> indices;
	copied_node_map = CopyNodesWithIndex(nodes_to_copy, indices, cursor);

	// replace all the arguments that use the copied nodes
	for (auto* arg : args_to_copy) {
		Node* from = arg->from_->get();
		if (!copied_node_map.contains(from)) {
			throw std::runtime_error("Optimize Kernels: Copy Fail");
		}
		Node* to = copied_node_map[from];
		arg->from_ = to->GetLable();
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
		unordered_set<Arg*> args_to_copy;
		unordered_set<Arg*> shape_args_to_copy;
		// go over all nodes in the kernel and check if their inputs can be copied
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			// go over all inputs
			for (auto& input : node->inputs_) {
				bool inside_kernel = input.from_->get()->HasParent(kernel);
				bool from_in_kernel = input.from_->get()->HasParent("kernel");

				if (!inside_kernel && !CannotCopyArgument(input))
				{
					// check if input is cheap enough to copy
					float input_cost = input.from_->get()->cost_;
					if (input_cost == -1.0) {
						throw std::runtime_error("Cost has not been computed");
					}
					bool cheap_enough = input_cost >= 0.0f && input_cost < MAX_KERNEL_COPY_COST;
					bool has_only_one_output = input.from_->get()->outputs_.size() == 1;
					if (cheap_enough || has_only_one_output) {
						args_to_copy.insert(&input);
					}
				}
				//shape arguments can not be inside kernels
				if (from_in_kernel && input.type_ == ArgType::Shape) {
					shape_args_to_copy.insert(&input);
				}
			}
		}

		//go over kernel shape arguments
		for (auto& arg : kernel->inputs_) {
			bool from_in_kernel = arg.from_->get()->HasParent("kernel");
			if (from_in_kernel && arg.type_ == ArgType::Shape) {
				shape_args_to_copy.insert(&arg);
			}
		}

		// copy the nodes that are outside the kernel inside
		CopyArguments(args_to_copy, kernel->child);
		// copy shape arguments before the kernel
		CopyArguments(shape_args_to_copy, kernel);
	}
}

#define MAX_LOAD_COPY 1024.0f

void IR::OptimizeKernelLoadOperations() {
	UpdateGraph();
	ComputeNodeCost();

	vector<Node*> kernels = GetNodesOfType("kernel");

	unordered_set<Node*> nodes_to_remove;

	for (auto kernel : kernels) {
		unordered_map<Node*, Node*> loads_to_copy;
		// go over all nodes in the kernel and check if their inputs can be copied
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			if (node->name != "load") continue;

			//get memory input
			Node* memory_input = node->GetArguments(ArgType::Memory)[0].from_->get();

			bool inside_kernel = memory_input->HasParent("kernel");

			bool cheap_enough = memory_input->cost_ >= 0.0f && memory_input->cost_ < MAX_LOAD_COPY;

			//if the memory input is used only once and is not a memory node
			if (cheap_enough && inside_kernel && memory_input->outputs_.size() == 1) {
				loads_to_copy[memory_input] = *node;
			}
		}

		for (auto load : loads_to_copy) {
			//get the load
			Node* memory_input = load.first;
			Node* load_node = load.second;

			//get the indices
			unordered_map<int, Node*> indices;
			for (auto& arg : load_node->inputs_) {
				if (arg.type_ == ArgType::Index) {
					indices[arg.index_] = arg.from_->get();
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

			//go over all outputs of the load node and replace them with the copied nodes
			for (auto& output : load_node->outputs_) {
				output->from_ = copied_node_map[memory_input]->GetLable();
			}

			//remove the load node since it is not needed anymore
			nodes_to_remove.insert(load_node);
		}
	}

	// remove the load nodes
	for (auto node : nodes_to_remove) {
		RemoveNode(node);
	}
}

#define MAX_HOST_COPY_COST 8192.0f

void IR::OptimizeHost() {
	ComputeNodeCost();

	//loop over all nodes and copy their arguments if they are cheap enough and inside kernels
	for (auto node = begin(); !node.end(); node.next()) {
		if (node->HasParent("kernel")) {
			continue;
		}

		unordered_set<Arg*> args_to_copy;
		// go over all inputs
		for (auto& input : node->inputs_) {
			bool inside_kernel = input.from_->get()->HasParent("kernel");

			if (inside_kernel && !CannotCopyArgument(input)) {
				// check if input is cheap enough to copy
				float input_cost = input.from_->get()->cost_;
				if (input_cost == -1.0) {
					throw std::runtime_error("Cost has not been computed");
				}
				bool cheap_enough = input_cost >= 0.0f && input_cost < MAX_HOST_COPY_COST;
				bool has_only_one_output = input.from_->get()->outputs_.size() == 1;

				if (cheap_enough || has_only_one_output) {
					args_to_copy.insert(&input);
				} else {
					throw std::runtime_error("Host optimization: Copy cost too high for node " + node->name + " with cost " + to_string(input_cost));
				}
			}
		}

		CopyArguments(args_to_copy, node.get());
	}
}



void IR::MoveShapeOutsideKernels() {
	UpdateGraph();

	// find all nodes that are used as shapes and are inside kernels
	map<Node*, Node*> nodes_to_copy;
	for (auto node = begin(); !node.end(); node.next()) {
		Node* kernel = node->GetParent("kernel");
		if (kernel == *node) continue;

		// go over all outputs arguments
		for (auto& output : node->outputs_) {
			if (output->type_ != ArgType::Shape) {
				continue;
			}

			// add the node to the set
			nodes_to_copy[node.get()] = kernel;
		}
	}

	for (auto [ node, kernel ] : nodes_to_copy) {
		//get all output arguments that are shapes
		unordered_set<Arg*> args_to_copy;
		int earliest_output_index = INT_MAX;
		Node* earliest_output = nullptr;
		for (auto& output : node->outputs_) {
			if (output->type_ == ArgType::Shape) {
				args_to_copy.insert(output);

				//get the earliest output
				if (output->index_ < earliest_output_index) {
					earliest_output_index = output->index_;
					earliest_output = output->to_->get();
				}
			}
		}

		Node* common_parent = earliest_output->GetCommonParent(kernel);

		// copy shape computation and put it before the earliest output (outside of the kernel if its inside)
		CopyArguments(args_to_copy, common_parent);
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
	for (auto node = begin(); !node.end(); node.next()) {
		//get node operation
		const string op = node->name;

		//get inputs
		map<int, const Tensor*> inputs = node->GetArgumentTensors(ArgType::Input);
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
			else if (op == "clamp")
			{
				// if all are constants, replace with result
				if (isConstant(inputs[0]) && isConstant(inputs[1]) && isConstant(inputs[2])) {
					// compute result
					result = ApplyFUNC(inputs[0], inputs[1], max);
					result = ApplyFUNC(result, inputs[2], min);
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
	return arg->type_ == ArgType::Memory &&
	       arg->to_->get()->op->HasAllTypes(OpType::Modifier);
}

void IR::RemoveUnusedOperations() {
	UpdateGraph();

	// use depth first search to find all nodes that are used for the output nodes
	unordered_set<Node*> used_nodes;

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
	for (auto node = begin(); !node.end(); node.next()) {
		if (node->memory_type_ == MemoryType::Output || node->op->HasAllTypes(OpType::Static)) {
			dfs(node.get());
		}
	}

	// remove all nodes that are not used
	unordered_set<Node*> nodes_to_remove;
	for (auto node = begin(); !node.end(); node.next()) {
		if (!used_nodes.contains(node.get())) {
			if (node->memory_type_ != MemoryType::Input && node->memory_type_ != MemoryType::Output)
			{
				nodes_to_remove.insert(node.get());
			}
			else
			{
				throw std::runtime_error("Input " + node->var_name + " is not being used");
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
		bool is_memory = node->op->HasAllTypes(OpType::Memory);
		unordered_map<Node*, float> input_costs;
		for (auto& input : node->inputs_) {
			if (input.type_ != ArgType::Memory &&
			    (input.type_ != ArgType::Shape && !is_memory)) {
				input_costs[input.from_->get()] = input.from_->get()->cost_;
			}
		}
		float input_cost = node->op->GetCost();
		for (auto& input : input_costs) {
			input_cost += abs(input.second);
		}
		node->cost_ = input_cost;
	}
}

map<Node*, vector<Arg*>> IR::GetKernelOutputs(Node* kernel)
{
	map<Node*, vector<Arg*>> node_output;
	for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
		bool is_output = node->memory_type_ == MemoryType::Output;
		vector<Arg*> outputs;

		for (auto& output : node->outputs_) {
			if (output->to_ == nullptr) continue;
			// if is a shape or memory argument, then skip (shape is loaded on CPU)
			if (output->type_ == ArgType::Shape) continue;
			Node* output_node = output->to_->get();
			if (!output_node->HasParent(kernel)) {
				outputs.push_back(output);
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
	for (auto& input : node->inputs_) {
		if (input.type_ == ArgType::Memory || input.type_ == ArgType::Shape)
			continue;

		Node* input_node = input.from_->get();
		bool is_in_a_kernel = input_node->HasParent("kernel");
		bool is_outside = !input_node->HasParent(kernel);
		bool is_memory = input_node->op->HasAllTypes(OpType::Memory);

		if (is_memory || (is_in_a_kernel && is_outside)) {
			// load the memory node before this node
			ExecuteExpressionBefore(node, [&]() {
				Tensor& loaded = Tensor::Load(*input_node->GetTensor(), indices, true);
				input.from_ = loaded.node_->GetLable();
			});
		}
	}
}

void IR::AddKernelGlobalLoadOperations() {
	// get kernels
	UpdateGraph();
	vector<Node*> kernels = GetNodesOfType("kernel");
	for (auto kernel : kernels) {

		// replace all inputs pointing to memory nodes with the memory node
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			AddNodeLoadOperations(node.get(), kernel, {});
		}
	}
}


void IR::AddMemoryOpIndices() {
	// get kernels
	UpdateGraph();
	vector<Node*> kernels = GetNodesOfType("kernel");
	for (auto kernel : kernels) {
		// get kernel shape arguments
		Arguments shape_args = kernel->GetArguments(ArgType::Shape);

		Tensors indices = Tensors();
		// add dimension index nodes
		ExecuteExpressionChild(kernel, [&]() {
			for (int i = 0; i < shape_args.size(); i++) {
				indices.push_back(&Tensor::Index(shape_args, i));
			}
		});
		int kernel_dim = (int)shape_args.size();

		// replace all inputs pointing to memory nodes with the memory node
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			if (!node->op->HasAllTypes(OpType::MemoryOp)) {
				continue;
			}

			Node* input_node = node->GetArguments(ArgType::Memory)[0].from_->get();
			map<int, const Tensor*> shape = input_node->GetArgumentTensors(ArgType::Shape);
			const Tensor* input_tensor = input_node->GetTensor();
			ArgMap index_args = node->GetArgumentMap(ArgType::Index);

			int memory_dim = (int)shape.size();
			ExecuteExpressionBefore(node.get(), [&]() {
				for (int i = 0; i < memory_dim; i++) {
					if (index_args.contains(i)) {
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
					node->AddArgument(index->node_, ArgType::Index, i);
				}
			});
		}
	}
}

void IR::AddKernelGlobalStoreOperations() {
	// get kernels
	UpdateGraph();
	vector<Node*> kernels = GetNodesOfType("kernel");

	// go over all outputs of each kernel and create memory nodes to store the
	// output
	for (auto kernel: kernels) {
		map<Node*, vector<Arg*>> node_output = GetKernelOutputs(kernel);

		for (auto out : node_output) {
			Node* output = out.first;
			// if the output is already a memory node, then skip
			if (output->op->HasAllTypes(OpType::Memory)) {
				continue;
			}

			Node* mem;
			// add memory node before this kernel
			ExecuteExpressionBefore(kernel, [&]() {
				mem = Tensor::Memory(output->GetArguments(ArgType::Shape), output->tensor_->type).node_;
				mem->debug_name = output->debug_name;

				if (output->memory_type_ == MemoryType::Output) {
					mem->memory_type_ = MemoryType::Output;
					mem->special_index_ = output->special_index_;
					output->memory_type_ = MemoryType::None;
				}
			});

			// go over all outputs of this node and replace their input with the
			// memory node
			for (auto& arg_out : node_output[output]) {
				if (arg_out->type_ != ArgType::Shape &&
				    arg_out->type_ != ArgType::Memory) {
					// if not a memory or shape argument, then the memory needs to be
					// loaded before the node
					ExecuteExpressionBefore(arg_out->to_->get(), [&]() {
						Tensor& loaded = Tensor::Load(*mem->GetTensor(), {}, true);
						// the node must now use the loaded value
						arg_out->from_ = loaded.node_->GetLable();
					});
				} else {
					// otherwise the memory can be used directly
					arg_out->from_ = mem->GetLable();
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
		bool is_memory = node->op->HasAllTypes(OpType::Memory);

		for (auto& input : node->inputs_) {
			if (input.type_ == ArgType::Memory ||
			    (input.type_ == ArgType::Shape && !is_memory))
				continue;

			if (input.from_->get()->op->HasAllTypes(OpType::Memory)) {
				// load the memory node before this node
				ExecuteExpressionBefore(node.get(), [&]() {
					Tensor& loaded = Tensor::Load(*input.from_->get()->GetTensor(), {}, true);
					input.from_ = loaded.node_->GetLable();
				});
			}
		}
	}
}

void IR::AddMemoryDeallocation()
{
	UpdateGraph();
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

			for (auto& output : node->outputs_)
			{
				Node* output_node = output->to_->get();
				if (output_node->op->HasAllTypes(OpType::MemoryReuse)) {
					dfs(output_node);
				} else {
					if (last_output_index < output_node->index_) {
						last_output_index = output_node->index_;
						last_output = output_node;
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
Tensor* ComputeFlatIndex(ArgMap memory_shape, vector<Tensor*> indices, map<int, const Tensor*> idx, int memory_dim, TensorIndexingMode mode = TensorIndexingMode::Clamp)
{
	if (memory_dim == 0)
	{
		return &Tensor::Constant(0);
	}

	int kernel_dim = (int)indices.size();

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
			int dim = node->GetTensor()->data[0];
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
			for (auto& input : node->inputs_) {
				if (input.from_->get()->name == "dim_id") {
					int dim = input.from_->get()->GetTensor()->data[0];
					if (dim >= dims) {
						throw runtime_error("Invalid dimension index " + to_string(dim) +
																							" for kernel of size " + to_string(dims));
					}

					// replace the dim node with the index node
					input.from_ = indices[dim]->node_->GetLable();
				}
			}
		}
	}

	// remove all dim nodes
	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}
}

void IR::LinearModeIndices(vector<Tensor*>& indices, Node* kernel, int dims, Tensors kernel_shape)
{
	Tensor* thread_index = nullptr;
	ExecuteExpressionChild(kernel, [&]() {
		thread_index = &kernel->GetTensor()->ThreadIndex();
		indices = ComputeIndicesFromLinearIndex(thread_index, kernel_shape, dims);
	});

	for (int i = 0; i < dims; i++) {
		indices[i]->SetDebugName("index_" + to_string(i));
	}
	
	ReplaceDimNodes(kernel, indices, dims);
}

void IR::MultiDimensionalModeIndices(vector<Tensor*>& indices, Node* kernel_, int dims, Tensors kernel_shape)
{
	//add dim_id nodes at the beginning of the kernel
	ExecuteExpressionChild(kernel_, [&]() {
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
	ExecuteExpressionChild(kernel_, [&]() {
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
	const Tensor* memory = node->GetArgumentTensors(ArgType::Memory)[0];

	ArgMap memory_shape = memory->node_->GetArgumentMap(ArgType::Shape);

	int memory_dim = MaxIndexCount(memory_shape);

	// get the index nodes
	map<int, const Tensor*> idx = node->GetArgumentTensors(ArgType::Index);

	if (idx.empty())
	{
		node->indexing_mode_ = TensorIndexingMode::Unsafe; //we can guarantee that the index is in bounds
	}

	Tensor* flat_index = ComputeFlatIndex(memory_shape, indices, idx, memory_dim, node->indexing_mode_);

	// TODO(Moroz): add different modes for clamping (e.g. clamp, wrap,
	// mirror, zero)

	// remove the index node edges
	node->RemoveArguments(ArgType::Index);

	// add the flat index node edge
	node->AddArgument(flat_index->node_, ArgType::Index, 0);
}

void IR::FinalizeMemoryIndexing() {
	UpdateGraph();
	vector<Node*> kernels = GetNodesOfType("kernel");

	vector<Tensor*> dispatch_checks;

	for (auto kernel : kernels) {
		Node* shape_node = kernel;
		if (shape_node == nullptr) continue;
		// load kernel shape
		map<int, const Tensor*> kernel_shape_map =
		    shape_node->GetArgumentTensors(ArgType::Shape);
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
			if (node->op->HasAllTypes(OpType::MemoryOp)) {
				ExecuteExpressionBefore(*node, [&]() { ComputeAddress(node.get(), indices); });
			}
		}
	}

	//now compute address for all nodes that are not in a kernel
	for (auto node = begin(); !node.end(); node.next()) {
		if (!node->HasParent("kernel") && node->op->HasAllTypes(OpType::MemoryOp)) {
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
	UpdateGraph();
	vector<Node*> kernels = GetNodesOfType("kernel");
	vector<Node*> nodes_to_remove;

	for (auto kernel : kernels) {
		// remove all kernel nodes that dont do anything
		int memory_modifiers = 0;
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			if (node->op->HasAllTypes(OpType::Modifier, OpType::MemoryOp)) {
				memory_modifiers++;
			}
			//if any output is outside the kernel, then the kernel is needed
			for (auto& output : node->outputs_) {
				if (!output->to_->get()->HasParent(kernel)) {
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
	Tensor* reduced = &Tensor::Constant(sum_shape, initial, array->type);
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
	if (array->type == DataType::Float) {
		float init = -FLT_MAX;
		initial = *(uint*)&init;
	}
	else if (array->type == DataType::Int) {
		int init = INT_MIN;
		initial = *(uint*)&init;
	}
	return ComputeReduction(
	    array, axis, [](Tensor* a, Tensor* b) { return &Tensor::max(*a, *b); },
	    "max", initial);
}

Tensor* ComputeMin(const Tensor* array, int axis) {
	uint initial = UINT_MAX;
	if (array->type == DataType::Float) {
		float init = FLT_MAX;
		initial = *(uint*)&init;
	}
	else if (array->type == DataType::Int) {
		int init = INT_MAX;
		initial = *(uint*)&init;
	}
	return ComputeReduction(
	    array, axis, [](Tensor* a, Tensor* b) { return &Tensor::min(*a, *b); },
	    "min", initial);
}

Tensor* ComputeProduct(const Tensor* array, int axis) {
	uint initial = 1;
	if (array->type == DataType::Float) {
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

Tensor* Transpose(const Tensor* array, map<int, int> permutation) {
	Tensors shape = array->GetShape();
	Tensors perm_shape = Tensors();
	for (int i = 0; i < (int)shape.size(); i++) {
		perm_shape.push_back(shape[permutation[i]]);
	}
	//create indices
	Tensors indices = Tensors();
	for (int i = 0; i < (int)shape.size(); i++) {
		indices.push_back(&Tensor::Index(perm_shape, i));
	}
	//permute indices to load the values
	Tensors perm_indices = Tensors();
	for (int i = 0; i < (int)shape.size(); i++) {
		perm_indices.push_back(indices[permutation[i]]);
	}
	Tensor& loaded = Tensor::Load(*array, perm_indices, true);
	loaded.SetDebugName("transposed");
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
	Tensors shape_a = a->GetShape();
	Tensors shape_b = b->GetShape();

	if (shape_a.size() < 2 || shape_b.size() < 2) {
		throw std::runtime_error("Matrix multiplication requires at least 2 dimensions");
	}

	//get shape of the result
	Tensors shape_c = Tensors();
	int dim_a = (int)shape_a.size();
	int dim_b = (int)shape_b.size();
	int max_dim = 0;
	Tensors max_shape = Tensors();
	//get the shape with most dimensions
	if (dim_a < dim_b) {
		max_dim = dim_b;
		max_shape = shape_b;
	} else {
		max_dim = dim_a;
		max_shape = shape_a;
	}

	for (int i = 0; i < max_dim - 2; i++) {
		shape_c.push_back(max_shape[i]);
	}
	shape_c.push_back(shape_a[dim_a - 2]);
	shape_c.push_back(shape_b[dim_b - 1]);

	const Tensor* sum_shape = shape_a[dim_a - 1];

	// get indices for c elements
	Tensors indices_c = Tensors();
	for (int i = 0; i < max_dim; i++) {
		indices_c.push_back(&Tensor::Index(shape_c, i));
	}

	// start with 0
	Tensor* c = &Tensor::Constant(shape_c, 0, a->type);
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
	UpdateGraph();
	// get all nodes for each type
	vector<Node*> nodes = GetNodesOfType(OpType::Algorithm);

	unordered_set<Node*> nodes_to_remove;

	// replace all nodes with the algorithmic primitive
	for (auto node : nodes) {
		//compute the sum after the node
		ExecuteExpressionAfter(node, [&]() {
			//get the input tensor
			map<int, const Tensor*> inputs = node->GetArgumentTensors(ArgType::Input);

			//get sum axis
			vector<int> axes;
			for (int i = 0; i < node->tensor_->data.size(); i++) {
				axes.push_back((int)node->tensor_->data[i]);
			}

			//compute the sum
			Tensor* result;
			//= ComputeSum(input, axis);
			if (node->name == "dim_sum") {
				result = ComputeSum(inputs[0], axes[0]);
			}
			else if (node->name == "dim_norm") {
				result = ComputeNorm(inputs[0], axes[0]);
			}
			else if (node->name == "dim_max") {
				result = ComputeMax(inputs[0], axes[0]);
			}
			else if (node->name == "dim_min") {
				result = ComputeMin(inputs[0], axes[0]);
			} 
			else if (node->name == "dim_mean") {
				result = ComputeMean(inputs[0], axes[0]);
			}
			else if (node->name == "dim_product") {
				result = ComputeProduct(inputs[0], axes[0]);
			}
			else if (node->name == "dim_any") {
				result = ComputeAny(inputs[0], axes[0]);
			}
			else if (node->name == "dim_all") {
				result = ComputeAll(inputs[0], axes[0]);
			}
			else if (node->name == "transpose") {
				//get the permutation
				int dim = (int)inputs[0]->GetDimension();
				map<int, int> permutation;
				for (int i = 0; i < dim; i++) {
					permutation[i] = i;
				}
				permutation[axes[0]] = axes[1];
				permutation[axes[1]] = axes[0];
				result = Transpose(inputs[0], permutation);
			}
			else if (node->name == "dot") {
				result = ComputeDot(inputs[0], inputs[1], axes[0]);
			}
			else if (node->name == "matmul") {
				result = ComputeMatMul(inputs[0], inputs[1]);
			}
			else {
				throw std::runtime_error("Unknown algorithmic primitive " + node->name);
			}

			//replace the node with the sum
			node->MakeOutputsUseGivenNode(result->node_);

			//copy over all memory flags to the new node
			//TODO make a function for this
			result->node_->memory_type_ = node->memory_type_;
			result->node_->special_index_ = node->special_index_;
		});

		//mark the node for removal
		nodes_to_remove.insert(node);
	}

	// remove all nodes that are not used
	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}
}

void IR::CompileIR() 
{
	// TODO (Moroz): Make sure that shape works with non-const tensors
	// TODO (Moroz): Add auto tests into build system

	CheckIR("Input", false, false);
	GetInputList();
	OptimizeOperations();
	CheckIR("Optimize operations", false, false);
	RemoveUnusedOperations();
	CheckIR("Remove Unused Operations 0", false, false);
	InsertAlgorithmicPrimitives();
	CheckIR("Insert Algorithmic Primitives", false, false);
	SeparateOperationsIntoKernels();
	CheckKernelShapes();

	CheckIR("Separate Operations Into Kernels", false, false);
	ReorderOperations();
	CheckIR("Reorder Operations", true, false);
	MoveShapeOutsideKernels();
	OptimizeKernels();
	OptimizeHost();
	CheckIR("Optimize kernels and host", true, false);
	for (int i = 0; i < 3; i++) {
		RemoveUnusedOperations();
		AddKernelGlobalLoadOperations();
		AddMemoryOpIndices();
		CheckIR("Load optimization 1 iteration " + to_string(i), true, false);
		OptimizeKernelLoadOperations();
		CheckIR("Load optimization 2 iteration " + to_string(i), true, false);
	}
	AddKernelGlobalStoreOperations();
	RemoveUnusedKernels();
	CheckIR("Add Kernel Global Memory Operations", true, true);
	AddMemoryOpIndices();
	ReorderOperations();
	OptimizeOperations();
	AddMemoryOpIndices();
	FinalizeMemoryIndexing();
	RemoveUnusedOperations();
	CheckIR("Finalize Memory Indexing", false, false);
	OptimizeKernels();
	OptimizeHost();
	RemoveUnusedOperations();
	CheckIR("Finalize Memory Indexing 2", true, true);
	RemoveUnusedKernels();
	OptimizeOperations();
	RemoveUnusedOperations();
	CheckIR("Remove Unused Operations 2", true, true);
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
		map<Node*, int> variables;
		map<Node*, int> memory_nodes;
		ArgMap shape = kernel->GetArgumentMap(ArgType::Shape);

		int variable_index = 0;
		int memory_index = 0;

		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			if (node->op->HasAllTypes(OpType::MemoryOp)) {
				// get the memory node
				const Tensor* memory = node->GetArgumentTensors(ArgType::Memory)[0];

				if (!memory_nodes.contains(memory->node_)) {
					memory_nodes[memory->node_] = memory_index++;
				}
			}

			// get all input arguments
			for (auto input : node->inputs_) {
				if (input.type_ == ArgType::Input)
				{
					Node* from = input.from_->get();
					bool from_outside_kernel = !from->HasParent(kernel);
					if (from_outside_kernel && !variables.contains(from)) {
						variables[from] = variable_index++;
					}
				}
			}
		}

		int dim = MaxIndexCount(shape);

		// add the kernel to the program
		program->AddKernel(kernel, variables, memory_nodes, shape, dim);
	}

	return program;
}

}  // namespace TensorFrost