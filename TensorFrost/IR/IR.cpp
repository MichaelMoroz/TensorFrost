#include "IR.h"

namespace TensorFrost {

int MaxIndexCount(ArgMap& map) {
	if (map.empty()) return 0;
	// maps are sorted by index, so the last element has the highest index
	return map.rbegin()->first + 1;
}

const Tensor *ArgumentManager::GetTensor(ArgType type, int index) {
	return Get(type, index)->GetTensor();
}

const Tensor & ArgumentManager::operator[](int index) {
	return *GetTensor(ArgType::Input, index);
}

void SwapLables(Node* a, Node* b) {
	// first swap the node addresses
	a->lable_->node_ = b;
	b->lable_->node_ = a;

	// now swap the labels
	Lable* temp = a->lable_;
	a->lable_ = b->lable_;
	b->lable_ = temp;
}

void CopyLable(Node* target, Node* copy) {
	// make old lable point to copy
	target->lable_->node_ = copy;
	// make new lable for target
	target->lable_ = new Lable(target);
}

Node* Node::GetLastChild() {
	Node* last = nullptr;
	for (NodeIterator it = NodeIterator(this); !it.end(); it.next()) {
		last = it.get();
	}
	return last;
}

inline bool KernelScope::IsBoundary(const Node* input, const Node* output,
                                    ArgType arg_type, bool is_identity) {
	const Operation* input_op = input->op;
	const Operation* output_op = output->op;

	// if this node loads something from another node, that node must not be in
	// this kernel
	if (output_op->HasAllTypes(OpClass::Load, OpClass::MemoryOp)) {
		return arg_type == ArgType::Memory &&
		       !is_identity;  // if its an identity load its fine
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
	Arguments indices = node->GetArguments(ArgType::Index);
	bool identity = indices.empty();

	for (auto& input : node->inputs_) {
		// get latest input version
		Node* latest = input.from_->get()->GetLastVersion(node);
		// check if input is the boundary of this kernel
		bool is_loop_boundary = latest->index_ > node->index_;
		if (IsBoundary(latest, node, input.type_, identity)) {
			if (is_loop_boundary) {
				latest = latest->GetParent("loop");
			}
			boundary_nodes.insert(latest);
		}
	}

	unordered_set<KernelScope*> child_scopes = ComputeScopes(node);

	int scope_count = (int)child_scopes.size();
	if (scope_count == 0) return;

	output_scopes.insert(child_scopes.begin(), child_scopes.end());

	if (scope_count == 1) {
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
	} else {
		// this node cant be in the scope if it has multiple child scopes
		begin = nullptr;
		end = nullptr;
	}
}

std::unordered_set<KernelScope*> KernelScope::ComputeScopes(Node* root) {
	std::unordered_set<KernelScope*> scopes;
	KernelScope* current_scope = new KernelScope();
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
		}
	}
	if (current_scope->IsValid()) {
		scopes.insert(current_scope);
	}
	return scopes;
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

Arg::~Arg() {
	// remove this arg from the output list of the from node
	if (from_) {
		if (from_->node_->outputs_.empty()) return;
		for (auto it = from_->node_->outputs_.begin();
		     it != from_->node_->outputs_.end(); ++it) {
			if (*it == this) {
				from_->node_->outputs_.erase(it);
				break;
			}
		}
	}
}

}  // namespace TensorFrost