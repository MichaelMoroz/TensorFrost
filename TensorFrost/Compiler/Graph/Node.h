#pragma once

#include "Compiler/Operations.h"
#include "Utility/Utility.h"
#include "Arguments.h"

namespace TensorFrost {

enum class NodeProp {
	Placeholder,
	Modified,
	IsStatic,
	OutputMemory,
	InputShapeDim,
	InputShapeMemory,
	InputMemory,
	InputMemoryList,
	KeepDims,
	DetachGrad,
	PassGrad,
	Count,
};

enum class MemoryType {
	None,
	Input,
	Output,
	Constant,
};

enum class IndexingMode {
	Unsafe,
	Clamp,
	Repeat,
	Mirror,
	Constant,
};

struct TFTypeDesc {
	TFType type;
	IndexingMode indexing_mode;
	uint constant_value;

};

string NodeFlagsToString(NodeProp flag);
using NodeProps = FlagSet<NodeProp, (int)NodeProp::Count>;

class Node {
	static int global_index;
 public:
	int index_ = -1;
	int debug_index = -1;
	string var_name = "";
	string debug_name;
	string name;
	float cost_ = -1.0f;

	Node *parent = nullptr, *child = nullptr, *next = nullptr, *prev = nullptr;
	const Operation* op;
	NodeProps flags;
	ArgumentManager args;
	const Tensor* tensor_;
	TFType type = TFType::Float;
	std::vector<uint> data;
	IndexingMode indexing_mode_; //clamp unless otherwise specified
	vector<int> group_size; //kernel properties


	Node(Node* prev = nullptr, Node* parent = nullptr) : parent(parent), prev(prev), args(this) {
		flags.set(NodeProp::Placeholder);
		debug_index = global_index++;
	}

    bool valid() {
        return !flags.has(NodeProp::Placeholder);
    }

	void UpdateEdges() {
		if (!child) child = new Node(nullptr, this);
		if (!next) next = new Node(this, parent);
		if (child->valid()) {
			child->parent = this;
		}
		if (next->valid()) {
			next->prev = this;
			next->parent = parent;
		}
	}

    //initialize and create next/child placeholders
    void initialize(Tensor* tensor, NodeArguments&& new_args, string&& new_name, TFType new_type, bool set_static = false) {
        if(valid()) {
            throw runtime_error("Node already initialized");
        }
		UpdateEdges();
        flags.remove(NodeProp::Placeholder);

		tensor_ = tensor;
		type = new_type;
		args.AddArguments(std::move(new_args));
		args.UpdateOutputs();
		flags.set(NodeProp::IsStatic, set_static);
		name = std::move(new_name);
		op = FindOperation(name);
		CheckNode();
		indexing_mode_ = IndexingMode::Clamp;
    }

	void CopyProperties(Node* other) {
		name = other->name;
		debug_name = other->debug_name;
		indexing_mode_ = other->indexing_mode_;
		group_size = other->group_size;
		type = other->type;

		flags.copy_all(other->flags);
	}

	void CopyMetadata(Node* other) {
		debug_name = other->debug_name;
		indexing_mode_ = other->indexing_mode_;
		group_size = other->group_size;

		flags.copy_all_except(other->flags, {NodeProp::Modified});
	}

	const Tensor* GetTensor() const;

	int ComputeDepth(Node* root = nullptr) const {
		int depth = 0;
		for (const Node* node = this; node != root; node = node->parent) {
			depth++;
		}
		return depth;
	}

	bool HasParent(Node* node)
	{
		for (Node* cur_parent = parent; cur_parent != nullptr; cur_parent = cur_parent->parent) {
			if (cur_parent == node) {
				return true;
			}
		}
		return false;
	}

	/// <summary>
	/// Make all outputs of this node use the given node as input, assuming that the output is further than min_index
	/// </summary>
	/// <param name="replacement"></param>
	/// <param name="min_index"></param>
	void MakeOutputsUseGivenNode(Node* replacement, int min_index = -1, bool make_modified = false) {
		for (auto [edge, to] : args.outputs_) {
			auto& [id, from] = edge;
			if (to->index_ >= min_index) {
				if(make_modified) {
					replacement->flags.set(NodeProp::Modified);
				}
				to->args.UpdateArgument(id, replacement);
			}
		}
	}

	Node* GetParent(string name)
	{
		for (Node* cur_parent = parent; cur_parent != nullptr; cur_parent = cur_parent->parent) {
			if (cur_parent->name == name) {
				return cur_parent;
			}
		}
		return this;
	}

	Node* GetChild(string name);

	//get the parent that has a common parent with another node
	Node* GetCommonParent(Node* other) {
		for (Node* cur_parent = this; cur_parent != nullptr; cur_parent = cur_parent->parent) {
			if (cur_parent->parent == other->parent) {
				return cur_parent;
			}
		}
		for (Node* cur_parent = other; cur_parent != nullptr; cur_parent = cur_parent->parent) {
			if (cur_parent->parent == this->parent) {
				return cur_parent;
			}
		}
		for (Node* cur_parent1 = this; cur_parent1 != nullptr;
		     cur_parent1 = cur_parent1->parent) {
			for (Node* cur_parent2 = other; cur_parent2 != nullptr;
			     cur_parent2 = cur_parent2->parent) {
				if (cur_parent1->parent == cur_parent2->parent) {
					return cur_parent1;
				}
			}
		}
		throw std::runtime_error("No common parent found");
	}

	Node* GetLastChild();

	//checks if the other node has all parents as this node
	bool HasCommonParents(Node* other) {
		for (Node* cur_parent = parent; cur_parent != nullptr; cur_parent = cur_parent->parent) {
			if (!other->HasParent(cur_parent)) {
				return false;
			}
		}
		return true;
	}

	bool HasParent(string name) {
		return GetParent(name) != this;
	}

	bool HasChild(string name) {
		return GetChild(name) != nullptr;
	}

	void SetMemoryType(NodeProp memory_type, int index = 0) {
		flags.set(memory_type, index);
	}

	void CheckNode() const {
		// must have operation
		if (op == nullptr) {
			throw std::runtime_error("Operation object not found");
		}

		// must have tensor
		if (tensor_ == nullptr && !flags.has(NodeProp::IsStatic)) {
			throw std::runtime_error("Tensor not found");
		}
	}

	Node* GetLastVersion(Node* latest_node) {
		//find last store/scatter operation
		Node* last_modifier = this;
		int last_index = -1;
		Node* loop_node = latest_node->GetParent("loop");
		bool has_loop = loop_node != latest_node;
		for (auto [edge, to] : args.outputs_) {
			auto& [id, from] = edge;
			bool is_memory = false;
			if (id.first != ArgType::Memory) {
				is_memory = true;
			}
			if (is_memory) {
				continue;
			}
			if (to->op->HasAllTypes(OpProp::Modifier)) {
				if (to->index_>last_index) {
					// either find the last modifier or the last memory node
					// or if there is a loop, find the last modifier inside the loop (i.e.
					// the previous iteration's modifier)
					// if the loop is scalar, then it doesn't matter
					bool before_latest = to->index_ < latest_node->index_;
					bool inside_loop = has_loop && to->HasParent(loop_node);
					bool not_same = to != latest_node;
					if ((before_latest || inside_loop) && not_same)
					{
						last_index = to->index_;
						last_modifier = to;
					}
				}
			}
		}
		return last_modifier;
	}

	Node* GetFinalVersion() {
		Node* final_version = this;
		int last_index = -1;
		for (auto [edge, to] : args.outputs_) {
			auto& [id, from] = edge;
			bool is_memory = false;
			if (id.first != ArgType::Memory) {
				is_memory = true;
			}
			if (is_memory) {
				continue;
			}
			if (to->op->HasAllTypes(OpProp::Modifier) && !to->op->HasAllTypes(OpProp::MemoryOp)) {
				if (to->index_ > last_index) {
					last_index = to->index_;
					final_version = to;
				}
			}
		}
		return final_version;
	}

	~Node();
};

//NodeIterator is a depth first iterator that iterates through the child nodes of a root node
class NodeIterator {
 public:
	Node* currentNode;
	Node* currentParent;
	Node* root;

#ifndef NDEBUG
	int iteration_count = 0;
	int parent_inconsistency_count = 0;
	unordered_set<Node*> visited;
	vector<Node*> path;
#endif

	NodeIterator() : currentNode(nullptr), root(nullptr), currentParent(nullptr) {}
	NodeIterator(Node* node, Node* root) : currentNode(node), root(root), currentParent(node->parent) {}
	NodeIterator(const Node* node, const Node* root)
	    : currentNode(const_cast<Node*>(node)), root(const_cast<Node*>(root)), currentParent(const_cast<Node*>(node->parent)) {}
	NodeIterator(Node* node_root)
	    : currentNode(node_root->child), root(node_root), currentParent(node_root) {}
	NodeIterator(const Node* node_root)
	    : currentNode(const_cast<Node*>(node_root->child)),
	      root(const_cast<Node*>(node_root)),
		  currentParent(const_cast<Node*>(node_root)) {}

	Node* operator*() const { return currentNode; }

	void update_current_node(Node* new_node) {
		currentNode = new_node;
#ifndef NDEBUG
		if (visited.contains(currentNode)) {
			throw std::runtime_error("Node already visited, potential cycle in operation graph");
		}
		visited.insert(currentNode);
		path.push_back(currentNode);
		iteration_count++;
#endif
	}

	NodeIterator& go_to_next() {
		if (!currentNode) {
			throw std::runtime_error("Invalid node");
		}

		update_current_node(currentNode->next);
#ifndef NDEBUG
		if (currentNode->parent != currentParent) {
			parent_inconsistency_count++;
		}
#endif

		return *this;
	}

	NodeIterator& go_to_parent() {
		if (!currentNode) {
			throw std::runtime_error("Invalid node");
		}

		update_current_node(currentNode->parent);
#ifndef NDEBUG
		currentParent = currentNode->parent;
#endif

		return *this;
	}

	NodeIterator& go_to_child() {
		if (!currentNode) {
			throw std::runtime_error("Invalid node");
		}

#ifndef NDEBUG
		currentParent = currentNode->parent;
#endif
		update_current_node(currentNode->child);

		return *this;
	}

	NodeIterator& up() {
		if (!currentNode) {
			throw std::runtime_error("Invalid node");
		}

		if (root == currentNode) {
			throw std::runtime_error("Already at root");
		}

		if (currentNode->parent != root) {
		    go_to_parent();
		} else {
			go_to_next();
		}
		return *this;
	}

	NodeIterator& forward() {
		if (!currentNode) {
			throw std::runtime_error("Invalid node");
		}

		if (!currentNode->valid()) {
			return *this;
		}

		if (!currentNode->next->valid()) {  // no next, try going up
			Node* parent = currentNode->parent;
			while (!parent->next->valid() && root != parent) {
				parent = parent->parent;
			}
			if (root != parent) {  // go to next sibling
				currentNode = parent;
			}
		}

		// just go to next node and stop if it's the end
		go_to_next();
		return *this;
	}

	// first child, then next
	NodeIterator& next() {
		if (!currentNode) {
			throw std::runtime_error("Invalid node");
		}

		if (!currentNode->valid()) {
			return *this;
		}

		if (currentNode->child->valid()) {  // has child, go down
			go_to_child();
			return *this;
		}

		forward();

		return *this;
	}

	bool end() { return !currentNode->valid(); }

	Node* operator->() { return currentNode; }

	Node* get() { return currentNode; }

	int depth() { return currentNode->ComputeDepth(root); }

	bool operator!=(const Node* node) { return currentNode != node; }
};


} // namespace TensorFrost