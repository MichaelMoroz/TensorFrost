#include "Scope.h"

namespace TensorFrost {

inline bool KernelScope::IsBoundary(const Node* input, const Node* output,
                                    ArgType arg_type, bool is_identity) {
	const Operation* input_op = input->op;
	const Operation* output_op = output->op;

	// if this node loads something from another node, that node must not be in
	// this kernel
	if (output_op->HasAllTypes(OpProp::Load, OpProp::MemoryOp)) {
		return arg_type == ArgType::Memory;
	}

	// if we are modifying memory, then the modified memory must not be in the
	// kernel
	if (output_op->HasAnyType(OpProp::Scatter, OpProp::Store) &&
	    !input_op->HasAnyType(OpProp::Scatter, OpProp::Store)) {
		return arg_type == ArgType::Memory;
	}

	//if the input is a load

	// shape should not be inside kernels
	if (arg_type == ArgType::Shape) {
		return true;
	}

	return false;
}

bool KernelScope::IsValid() const {
	// check if the scope is valid
	if (begin == nullptr || end == nullptr) return false;

	// begin and end must have the same parent
	if (begin->parent != end->parent) return false;

	if (begin->index_ < 0 || end->index_ < 0)
		throw std::runtime_error("Indices are not computed");

	// begin must be before or at the same index as end
	if (begin->index_ > end->index_) return false;

	// check if the boundary nodes are not in the scope
	for (Node* boundary_node : boundary_nodes) {
		if (boundary_node->index_ >= begin->index_ &&
		    boundary_node->index_ <= end->index_) {
			return false;
		}
	}

	return true;
}

KernelScope::KernelScope(Node* node,
                         unordered_set<KernelScope*>& output_scopes) : begin(node), end(node) {
	scope_shape = ShapeInfo(node);

	// if host only, then this can not be a valid kernel scope
	if (node->op->HasAllTypes(OpProp::HostOnly)) {
		begin = nullptr;
		end = nullptr;
		return;
	}

	// find boundary nodes
	bool identity = node->args.Count(ArgType::Index) == 0;

	for (auto& input : node->args.inputs_) {
		// get latest input version
		Node* latest = input.second->GetLastVersion(node);
		// check if input is the boundary of this kernel
		bool is_loop_boundary = latest->index_ > node->index_;
		if (IsBoundary(latest, node, input.first.first, identity)) {
			if (is_loop_boundary) {
				latest = latest->GetParent("loop");
			}
			boundary_nodes.insert(latest);
		}
	}

	pair<unordered_set<KernelScope*>, bool> all_scopes = ComputeScopes(node);
	auto child_scopes = all_scopes.first;
	bool host_only = all_scopes.second;

	if(host_only) {
		begin = nullptr;
		end = nullptr;
		return;
	}

	int scope_count = (int)child_scopes.size();
	if (scope_count == 0) return;

	output_scopes.insert(child_scopes.begin(), child_scopes.end());

	//if there is more than one child scope, then this node can not be in the scope
	if (scope_count > 1) {
		begin = nullptr;
		end = nullptr;
		return;
	}

	KernelScope* child_scope = *child_scopes.begin();
	AddBoundaryNodes(child_scope->boundary_nodes);

	ShapeCompareResult result =
	    CompareShape(scope_shape, child_scope->scope_shape, true);

	if (result.compatible) {
		scope_shape = result.broadcast_shape;
	} else {
		throw std::runtime_error("Something went wrong");
		// this node cant be in the scope if it has incompatible shapes
		begin = nullptr;
		end = nullptr;
	}

}

pair<std::unordered_set<KernelScope *>, bool> KernelScope::ComputeScopes(Node *root) {
	std::unordered_set<KernelScope*> scopes;
	KernelScope* current_scope = new KernelScope();
	bool host_only = false;
	for (auto node = NodeIterator(root); !node.end(); node.go_to_next()) {
		std::unordered_set<KernelScope*> child_scopes;
		KernelScope* node_scope = new KernelScope(node.get(), child_scopes);
		if (node_scope->IsValid()) {  // can be merged
			KernelScope* merged = KernelScope::Merge(current_scope, node_scope);
			if (merged->IsValid()) {
				current_scope = merged;
			} else {
				if (current_scope->IsValid()) {
					scopes.insert(current_scope);
				}
				current_scope = node_scope;
			}
		} else { // has child kernels
			// add all child scopes
			scopes.insert(child_scopes.begin(), child_scopes.end());
			// add current scope
			if (current_scope->IsValid()) {
				scopes.insert(current_scope);
			}
			// create a new empty scope
			current_scope = new KernelScope();
			host_only = true;
		}
	}
	if (current_scope->IsValid()) {
		scopes.insert(current_scope);
	}
	return {scopes, host_only};
}

KernelScope* KernelScope::Merge(KernelScope* a, KernelScope* b) {
	bool a_valid = a->IsValid();
	bool b_valid = b->IsValid();

	if (!a_valid && !b_valid) 
		throw std::runtime_error("Invalid kernel scopes for merging");

	if (!a_valid) return b;
	if (!b_valid) return a;

	if (a->end->next != b->begin)
		throw std::runtime_error("Trying to merge non-adjacent kernel scopes");

	ShapeCompareResult result = CompareShape(a->scope_shape, b->scope_shape, true);

	if (!result.compatible) return new KernelScope();

	KernelScope* new_scope = new KernelScope(a->begin, b->end, result.broadcast_shape, a->boundary_nodes);
	new_scope->AddBoundaryNodes(b->boundary_nodes);

	if (!new_scope->IsValid()) return new KernelScope();

	return new_scope;
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
				if(res.a_dim != -1 && res.b_dim != -1) {
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

}  // namespace TensorFrost