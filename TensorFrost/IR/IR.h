#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <stack>

#include "Operations.h"
#include "Utility/Utility.h"

namespace TensorFrost {
class Tensor;
class Node;

class Lable {
 public:
	Node* node_;
	explicit Lable(Node* node) : node_(node) {}

	Node* operator*() const { return node_; }
	Node* operator->() const { return node_; }
	[[nodiscard]] Node* get() const { return node_; }
};

class Arg {
 public:
	enum Type {
		Input,
		Index,
		Shape,
		Memory,
		None,
	};

	static inline const map<Type, string> type_names = {
	    {Type::Input, "Input"}, {Type::Index, "Index"}, {Type::Shape, "Shape"},
		{Type::Memory, "Memory"}, {Type::None, "None"},
	};

	static string TypeToString(Type type) { return type_names.at(type); }

	Type type_;
	Lable* from_;
	Lable* to_{nullptr};
	int index_;

	Arg(Type type, Lable* node, int index)
	    : type_(type), from_(node), index_(index) {}

	void SetOutput(Lable* output) { to_ = output; }
};

using ArgMap = map<int, const Arg*>;
using Arguments = vector<Arg>;
using ArgumentRefs = vector<const Arg*>;
using Tensors = vector<const Tensor*>;

int MaxIndexCount(ArgMap& map);

enum class MemoryType {
	None,
	Input,
	Output,
	Shape,
	Constant,
};

class Node {
 public:
	string var_name = "none";
	string name;
	float cost_ = -1.0f;
	
	Node *parent, *child, *prev, *next;
    bool placeholder;
	
	const Operation* op;
	const Tensor* tensor_;

	Lable* lable_ = nullptr;

	Arguments inputs_;
	vector<Arg*> outputs_;
	MemoryType memory_type_ = MemoryType::None;
	int memory_index_ = 0;
	int global_index_ = 0;

	bool has_been_modified_ = false;
	bool is_static = false;

	Node(Node* prev = nullptr, Node* parent = nullptr) : child(nullptr), next(nullptr), placeholder(true), parent(parent), prev(prev) {}

    bool valid() {
        return !placeholder;
    }

    //initialize and create next/child placeholders
    void initialize(Tensor* tensor, Arguments&& new_args, string&& new_name, bool set_static = false) {
        if(valid()) {
            throw runtime_error("Node already initialized");
        }
        if(!child) child = new Node(nullptr, this);
        if(!next) next = new Node(this, parent);
        placeholder = false;

		tensor_ = tensor;
		    inputs_ = std::move(new_args);
		name = std::move(new_name);
		lable_ = new Lable(this);
		is_static = set_static;
		UpdateArgumentOutputs();
		op = &FindOperation(name);
		CheckClustering();
    }

	[[nodiscard]] const Tensor* GetTensor() const;

	[[nodiscard]] Lable* GetLable() const { return lable_; }

	void SetAsModified()
	{
		has_been_modified_ = true;
	}

	[[nodiscard]] bool HasBeenModified() const
	{
		return has_been_modified_;
	}

	void UpdateArgumentOutputs() {
		for (Arg& input : inputs_) {
			input.SetOutput(lable_);
		}
	}

	void UpdateOutputs() {
		for (auto& input : inputs_) {
			input.SetOutput(GetLable());
			input.from_->get()->outputs_.push_back(&input);
		}
	}

	void SetMemoryType(MemoryType memory_type, int index = 0) {
		memory_type_ = memory_type;
		memory_index_ = index;
	}

	int MaxIndex(Arg::Type type) const {
		int max_index = 0;
		for (const auto& input : inputs_) {
			if (input.type_ == type) {
				max_index = std::max(max_index, input.index_);
			}
		}
		return max_index;
	}

	[[nodiscard]] Arguments GetArguments(Arg::Type type) const {
		Arguments result = Arguments();
		for (const auto& input : inputs_) {
			if (input.type_ == type) {
				result.push_back(input);
			}
		}
		// sort by index
		std::sort(result.begin(), result.end(),
		          [](const Arg& a, const Arg& b) {
			          return a.index_ < b.index_;
		          });
		return result;
	}

	ArgMap GetArgumentMap(Arg::Type type) const {
		ArgMap result = ArgMap();
		for (auto& input : inputs_) {
			if (input.type_ == type) {
				result[input.index_] = &input;
			}
		}
		return result;
	}

	[[nodiscard]] map<int, const Tensor*> GetArgumentTensors(
	    Arg::Type type) const {
		// get the arguments
		Arguments arguments = GetArguments(type);
		// convert to tensors
		map<int, const Tensor*> result = map<int, const Tensor*>();
		for (auto& argument : arguments) {
			result[argument.index_] = argument.from_->node_->GetTensor();
		}
		return result;
	}

	void RemoveArguments(Arg::Type type) {
		for (auto it = inputs_.begin(); it != inputs_.end();) {
			if (it->type_ == type) {
				it = inputs_.erase(it);
			} else {
				++it;
			}
		}
	}

	void AddArgument(Node* node, Arg::Type type, int index = 0) {
		inputs_.emplace_back(type, node->GetLable(), index);
	}

	void CheckClustering() const {
		// must have operation
		if (op == nullptr) {
			throw std::runtime_error("Operation object not found");
		}

		// must have tensor
		if (tensor_ == nullptr && !is_static) {
			throw std::runtime_error("Tensor not found");
		}
	}

	const Node* GetLastVersion(const Node* latest_node) const {
		//find last store/scatter operation
		const Node* last_modifier = this;
		int last_index = -1;
		for (auto& output : outputs_) {
			if (output->type_ != Arg::Type::Memory) {
				continue;
			}
			Node* output_node = output->to_->get();
			if (output_node->op->HasAllTypes(OpType::Modifier, OpType::MemoryOp)) {
				if (output_node->global_index_ > last_index &&
				    output_node->global_index_ < latest_node->global_index_) {
					last_index = output_node->global_index_;
					last_modifier = output_node;
				}
			}
		}
		return last_modifier;
	}

	~Node();
};

void SwapLables(Node* a, Node* b);
void CopyLable(Node* target, Node* copy);

enum class ScopeType {
	None,
	Host,
	Kernel,
	HostLoop,
	KernelLoop,
};


ScopeType GetScopeType(const Node* node);

class ClusterProp {
 public:
	vector<Node*> clusters;
	map<Node*, vector<Node*>> output;
	map<Node*, vector<Arg*>> node_output;

	ClusterProp(map<Node*, vector<Node*>> cluster_outputs,
	            map<Node*, vector<Arg*>> output, vector<Node*> clusters)
	    : output(std::move(cluster_outputs)),
	      node_output(std::move(output)),
	      clusters(std::move(clusters)) {}
};

enum class KernelIndexingMode
{
	Linear,
	MultiDimensional,
	LinearBlocks,
	MultiDimensionalBlocks,
};

enum class TensorIndexingMode {
	Unsafe,
	Clamp,
	Repeat,
	Zero,
};


class NodeIterator {
public:
    Node* currentNode;
	Node* root;

	NodeIterator() : currentNode(nullptr), root(nullptr) {}
    NodeIterator(Node* node, Node* root) : currentNode(node), root(root) {}
	NodeIterator(const Node* node, const Node* root)
	    : currentNode(const_cast<Node*>(node)),
	      root(const_cast<Node*>(root)) {}
	NodeIterator(Node* node_root) : currentNode(node_root->child), root(node_root) {}
	NodeIterator(const Node* node_root)
	    : currentNode(const_cast<Node*>(node_root->child)),
	      root(const_cast<Node*>(node_root)) {}

    Node* operator*() const {
        return currentNode;
    }

    //first child, then next
	NodeIterator& next() {
        if(!currentNode->valid()) {
            return *this;
        }

        if (currentNode->child->valid()) { //has child, go down
            currentNode = currentNode->child;
            return *this;
        }
        
        if (!currentNode->next->valid()) { //no next, try going up
			Node* parent = currentNode->parent;
			while (!parent->next->valid() && root != parent) {
				parent = parent->parent;
			}
			if (root != parent) { //go to next sibling
				currentNode = parent;
			}
        }

        currentNode = currentNode->next;
        return *this;
    }

    bool end() {
        return !currentNode->valid();
    }

    Node* operator->() {
        return currentNode;
    }

	Node* get() { return currentNode; }

	int depth() {
		int depth = 0;
		Node* node = currentNode;
		while (node->parent != root) {
			node = node->parent;
			depth++;
		}
		return depth;
	}
};

class IR {
public:
	Node* root;
    NodeIterator cursor;

	IR() {
        root = new Node();
        root->initialize(nullptr, {}, "host", true);
        cursor = NodeIterator(root);
    }

    ~IR() {
        vector<Node*> to_delete;
		for (auto node = begin(); !node.end(); node.next()) {
			to_delete.push_back(*node);
        }
        for (Node* node : to_delete) {
            delete node;
        }
		delete root;
    }

    NodeIterator begin() const {
        return NodeIterator(root);
    }

    Node* AddNode(Tensor* tensor, Arguments&& args, string&& name) {
        if (cursor->valid()) { //already initialized, add new node before cursor
            Node* newNode = new Node(cursor->prev, cursor->parent);
            cursor->prev->next = newNode;
            cursor->prev = newNode;
            newNode->initialize(tensor, std::move(args), std::move(name));
			return newNode;
        } else {
            cursor->initialize(tensor, std::move(args), std::move(name));
            cursor.next();
			return cursor->prev;
        }
    }

    void RemoveNode(Node* node) {
        if (node->valid()) {
            //if direct child of its parent
            if (node->parent && node->parent->child == node) {
                node->parent->child = node->next;
            } else if (node->prev) {
                node->prev->next = node->next;
            }

            node->next->prev = node->prev;
            
            delete node;
        }
    }

    void SetCursor(Node* node) {
        cursor = NodeIterator(node, root);
    }

    void ExecuteExpressionAfter(Node* node, const function<void()>&& expression) {
        NodeIterator oldCursor = cursor;
        SetCursor(node->next);
        expression();
        cursor = oldCursor;
    }

    void ExecuteExpressionBefore(Node* node, const function<void()>&& expression) {
        NodeIterator oldCursor = cursor;
        SetCursor(node);
        expression();
        cursor = oldCursor;
    }

	void ExecuteExpressionChild(Node* node, const function<void()>&& expression) {
		NodeIterator oldCursor = cursor;
		SetCursor(node->child);
		expression();
		cursor = oldCursor;
	}

    void moveNodeTo(Node* node, Node* new_prev)
    {
        if (node->valid()) {
            if (node->parent->child == node) {
                node->parent->child = node->next;
            } else {
                node->prev->next = node->next;
            }

            node->next->prev = node->prev;

            node->prev = new_prev;
            node->next = new_prev->next;
            new_prev->next->prev = node;
            new_prev->next = node;
        }
    }

	[[nodiscard]] map<Node*, Node*> CopyComputation(
	    const unordered_set<Node*>& targets) const;

	void RecomputeGlobalIndices() const;
	void CheckIR(string name, bool check_clustering, bool check_kernels) const;
	void PrintListing(string name, bool compact, map<Node*, string> invalid_nodes) const;
	void GetInputList();
	void GetOutputList();
	void ComputeStatistics();
	//void ReorderOperations();
	//void OptimizeKernels();
	void OptimizeOperations();
	void RemoveUnusedOperations();
	void SeparateOperationsIntoKernels() const;
	void UpdateNodeOutputs() const;
	//[[nodiscard]] ClusterProp GetClusterProperties() const;
	//void AddKernelGlobalMemoryOperations();
	//void LinearModeIndices(Tensor*& thread_index, vector<Tensor*>& indices,
	//                       Node* cluster, int dims, Tensors kernel_shape);
	//void MultiDimensionalModeIndices(Tensor*& thread_index,
	//                                 vector<Tensor*>& indices, Node* kernel_,
	//                                 int dims, Tensors kernel_shape);
	//void FinalizeMemoryIndexing();
	void CompileIR();

	//TODO (Moroz): Make this per kernel
	void SetKernelIndexingMode(KernelIndexingMode indexing_mode)
	{
		indexing_mode_ = indexing_mode; 
	}

	//TODO (Moroz): Make this per tensor
	void SetTensorIndexingMode(TensorIndexingMode indexing_mode)
	{
		tensor_indexing_mode_ = indexing_mode;
	}

	int input_memory_count = 0;
	int output_memory_count = 0;
	int temp_memory_count = 0;

	vector<Node*> memory_inputs;
	unordered_map<Node*, unordered_map<int, Node*>> shape_memory_map;
	unordered_map<int, Node*> output_memory_map;
	KernelIndexingMode indexing_mode_ = KernelIndexingMode::Linear;
	TensorIndexingMode tensor_indexing_mode_ = TensorIndexingMode::Unsafe;
};
struct ShapeCompareResult {
	bool compatible;
	bool is_broadcast;
	int a_dim;
	int b_dim;
	int min_dim;
};

ShapeCompareResult CompareShape(const Node* a, const Node* b);

}  // namespace TensorFrost