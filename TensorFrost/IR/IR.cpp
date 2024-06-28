#include "IR.h"

namespace TensorFrost {

void ArgumentManager::UpdateOutputs() {
	for (auto& [id, node] : inputs_) {
		node->args.AddOutput(id, node_);
	}
}

void ArgumentManager::ClearOutputs() {
	outputs_.clear();
}

const Tensor *ArgumentManager::GetTensor(ArgType type, int index) const {
	return Get(type, index)->GetTensor();
}

const Tensor & ArgumentManager::operator[](int index) const {
	return *GetTensor(ArgType::Input, index);
}

map<int, const Tensor *> ArgumentManager::GetTensors(ArgType type) const {
	map<int, const Tensor *> tensors;
	for (auto& [id, node] : inputs_) {
		if (id.first == type) {
			tensors[id.second] = node->GetTensor();
		}
	}
	return tensors;
}

ArgumentManager::~ArgumentManager() {

}

bool ArgumentManager::CannotMoveArgument(ArgID id) {
	Node* from = inputs_[id];
	Node* to = node_;
	return (id.first == ArgType::Memory &&
	        !to->op->HasAllTypes(OpClass::Set)) ||
	       (id.first  == ArgType::Shape && !to->op->HasAllTypes(OpClass::Memory)) ||
	       from->op->HasAllTypes(OpClass::Memory) ||
	       (from->name == "const" && to->op->HasAllTypes(OpClass::Memory)); //FIX THIS
}

bool ArgumentManager::CannotCopyArgument(ArgID id) {
	Node* from = inputs_[id];
	Node* to = node_;
	bool shape = id.first == ArgType::Shape;
	bool to_memory = to->op->HasAllTypes(OpClass::Memory);
	bool shape_not_memory = shape && !to_memory;
	return id.first == ArgType::Memory || shape_not_memory ||
	       from->op->HasAllTypes(OpClass::Static) ||
	       from->op->HasAllTypes(OpClass::Memory) || from->HasBeenModified();
}

bool ArgumentManager::IsChangingInput(ArgID arg) {
	return arg.first == ArgType::Memory &&
	       node_->op->HasAllTypes(OpClass::Modifier);
}


Node* Node::GetLastChild() {
	NodeIterator it = NodeIterator(this);
	for (; !it.end(); it.go_to_next()) {}
	return it.get();
}

inline bool KernelScope::IsBoundary(const Node* input, const Node* output,
                                    ArgType arg_type, bool is_identity) {
	const Operation* input_op = input->op;
	const Operation* output_op = output->op;

	// if this node loads something from another node, that node must not be in
	// this kernel
	if (output_op->HasAllTypes(OpClass::Load, OpClass::MemoryOp)) {
		return arg_type == ArgType::Memory;
	}

	// if we are modifying memory, then the modified memory must not be in the
	// kernel
	if (output_op->HasAnyType(OpClass::Scatter, OpClass::Store) &&
	    !input_op->HasAnyType(OpClass::Scatter, OpClass::Store)) {
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
	if (node->op->HasAllTypes(OpClass::HostOnly)) {
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

const map<ArgType, string> arg_type_names = {
	{ArgType::Input, "Input"}, {ArgType::Index, "Index"}, {ArgType::Shape, "Shape"},
	{ArgType::Memory, "Memory"}, {ArgType::None, "None"},
};

string TypeToString(ArgType type) {
	return arg_type_names.at(type);
}

}  // namespace TensorFrost