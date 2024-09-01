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
	NoLoadFusion,
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
	Mirror
};

struct TFTypeDesc {
	TFType type;
	IndexingMode indexing_mode;
	uint constant_value;

};

string NodeFlagsToString(NodeProp flag);
string IndexingModeToString(IndexingMode mode);
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

#ifndef NDEBUG
	string created_in;
#endif

	Node(Node* prev = nullptr, Node* parent = nullptr) : parent(parent), prev(prev), args(this) {
		flags.set(NodeProp::Placeholder);
		debug_index = global_index++;
		indexing_mode_ = IndexingMode::Clamp;
	}

    bool valid() {
        return !flags.has(NodeProp::Placeholder);
    }

	void UpdateEdges();

	//initialize and create next/child placeholders
    void initialize(Tensor* tensor, NodeArguments&& new_args, string&& new_name, TFType new_type, bool set_static = false);

	void CopyProperties(Node* other);
	void CopyMetadata(Node* other);

	const Tensor* GetTensor() const;
	int ComputeDepth(Node* root = nullptr) const;
	bool HasParent(Node* node) const;

	/// <summary>
	/// Make all outputs of this node use the given node as input, assuming that the output is further than min_index
	/// </summary>
	/// <param name="replacement"></param>
	/// <param name="min_index"></param>
	void ReplaceThisWithGivenNode(Node* replacement, int min_index = -1, bool make_modified = false, bool copy_metadata = true);

	Node* GetParent(string name);
	Node* GetChild(string name);

	//get the parent that has a common parent with another node
	Node* GetCommonParent(Node* other);

	Node* GetLastChild();

	//checks if the other node has all parents as this node
	bool HasCommonParents(Node* other) const;

	bool HasParent(string name);
	bool HasChild(string name);

	void SetMemoryType(NodeProp memory_type, int index = 0);
	void CheckNode() const;
	Node* GetLastVersion(Node* latest_node);
	Node* GetFinalVersion();

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