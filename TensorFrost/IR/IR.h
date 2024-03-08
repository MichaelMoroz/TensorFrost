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
class Scope;

enum class ScopeType {
	None,
	Host,
	Kernel,
};


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
	int index_ = 0;
	string var_name = "none";
	string name;
	float cost_ = -1.0f;
	
	Node *parent, *child, *prev, *next;
    bool placeholder;

	//only true after graph has been updated
	Node *true_prev, *true_next;
	
	const Operation* op;
	const Tensor* tensor_;

	ScopeType scope_type_ = ScopeType::None;

	Lable* lable_ = nullptr;

	Arguments inputs_;
	vector<Arg*> outputs_;
	MemoryType memory_type_ = MemoryType::None;
	int memory_index_ = 0;

	bool has_been_modified_ = false;
	bool is_static = false;

	Node(Node* prev = nullptr, Node* parent = nullptr) : child(nullptr), next(nullptr), placeholder(true), parent(parent), prev(prev) {}

    bool valid() {
        return !placeholder;
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
    void initialize(Tensor* tensor, Arguments&& new_args, string&& new_name, bool set_static = false) {
        if(valid()) {
            throw runtime_error("Node already initialized");
        }
		UpdateEdges();
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


	Node* GetParent(string name)
	{
		for (Node* cur_parent = parent; cur_parent != nullptr; cur_parent = cur_parent->parent) {
			if (cur_parent->name == name) {
				return cur_parent;
			}
		}
		return this;
	}

	Node* GetLastChild() {
		Node* last_child = child;
		while (last_child->next->valid()) {
			last_child = last_child->next;
		}
		return last_child;
	}

	bool HasParent(string name) {
		return GetParent(name) != this;
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
				if (output_node->index_ > last_index &&
				    output_node->index_ < latest_node->index_) {
					last_index = output_node->index_;
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


class NodeIterator {
 public:
	Node* currentNode;
	Node* root;

	NodeIterator() : currentNode(nullptr), root(nullptr) {}
	NodeIterator(Node* node, Node* root) : currentNode(node), root(root) {}
	NodeIterator(const Node* node, const Node* root)
	    : currentNode(const_cast<Node*>(node)), root(const_cast<Node*>(root)) {}
	NodeIterator(Node* node_root)
	    : currentNode(node_root->child), root(node_root) {}
	NodeIterator(const Node* node_root)
	    : currentNode(const_cast<Node*>(node_root->child)),
	      root(const_cast<Node*>(node_root)) {}

	Node* operator*() const { return currentNode; }

	NodeIterator& forward() {
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
		currentNode = currentNode->next;
		return *this;
	}

	// first child, then next
	NodeIterator& next() {
		if (!currentNode->valid()) {
			return *this;
		}

		if (currentNode->child->valid()) {  // has child, go down
			currentNode = currentNode->child;
			return *this;
		}

		forward();

		return *this;
	}

	NodeIterator& true_next() {
		currentNode = currentNode->true_next;
		return *this;
	}

	NodeIterator& true_prev() {
		currentNode = currentNode->true_prev;
		return *this;
	}

	bool end() { return !currentNode->valid(); }

	Node* operator->() { return currentNode; }

	Node* get() { return currentNode; }

	int depth() { return currentNode->ComputeDepth(root); }

	bool operator!=(const Node* node) { return currentNode != node; }
};

ScopeType GetScopeType(const Node* node);

class Scope
{
 public:
	Node* begin;
	Node* end;
	Node* shape_node;
	ScopeType type = ScopeType::None;
	int shape_dim = 0;

	Scope(Node* begin) 
		: begin(begin), end(begin), shape_node(begin) { UpdateEnd(begin); }

	Scope(Node* begin, Node* end)
	    : begin(begin), end(end) { RecomputeScope(); }

	bool InScope(const Node* node) {
		int begin_id = begin->index_;
		int end_id = end->index_;
		int node_id = node->index_;
		return node_id >= begin_id && node_id <= end_id;
	}

	void UpdateType(Node* node) {
		// if the end node is a memory node, it must be on the cpu
		if (node->name == "memory") {
			if (type == ScopeType::Kernel) {
				throw std::runtime_error("Memory node in kernel scope");
			}
			type = ScopeType::Host;
		} else if (shape_dim > 0 || node->scope_type_ == ScopeType::Kernel) { // non-scalars must be in a kernel
			if (type == ScopeType::Host) {
				throw std::runtime_error("Kernel node in host scope");
			}
			type = ScopeType::Kernel;
		}
	}

	void UpdateEnd(Node* new_end) {
		end = new_end;
		UpdateShape(end);
		UpdateType(end);
	}

	void UpdateShape(Node* node)
	{
		ArgMap shape = node->GetArgumentMap(Arg::Type::Shape);
		int dim = MaxIndexCount(shape);
		if (node->name == "memory") dim = 0;
		if (dim >= shape_dim) {
			shape_dim = dim;
			shape_node = node;
		}
	}

	void RecomputeScope()
	{
		shape_dim = 0;
		type = ScopeType::None;
		for (auto node = NodeIterator(begin, begin);
		     node->index_ <= end->index_; node.true_next()) {
			UpdateShape(*node);
			UpdateType(*node);
		}
	}

	static vector<Scope*> GetScopes(Node* begin, Node* end) {
		int begin_depth = begin->ComputeDepth();
		int end_depth = end->ComputeDepth();
		vector<Scope*> scopes;
		if (begin_depth <= end_depth) {
			Node* prev_parent = nullptr;
			for (Node* cur_parent = end; cur_parent != begin->parent;
			     cur_parent = cur_parent->parent) {
				if (prev_parent && prev_parent->prev && prev_parent->prev->valid()) {
					scopes.push_back(new Scope(cur_parent->child, prev_parent->prev));
				}
				prev_parent = cur_parent;
			}

			if (prev_parent && prev_parent->prev && prev_parent->prev->valid()) {
				scopes.push_back(new Scope(begin, prev_parent->prev));
			}
		}
		else
		{
			throw std::runtime_error("Invalid scope");
			//if (begin->parent->next) {
			//	scopes.push_back(new Scope(begin, begin->parent->next->true_prev));
			//	if (begin->parent->next != end)
			//	{
			//		scopes.push_back(new Scope(begin->parent->next, end->prev));
			//	}
			//}
			
			//Node* prev_parent = nullptr;
			//for (Node* cur_parent = begin->parent; cur_parent != end->parent;
			//     cur_parent = cur_parent->parent) {
			//	if (prev_parent && prev_parent->prev && prev_parent->prev->valid()) {
			//		scopes.push_back(new Scope(prev_parent->prev->next, cur_parent->child));
			//	}
			//	prev_parent = cur_parent;
			//}
		}
		

		return scopes;
	}
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
			if (cursor->prev) 
				cursor->prev->next = newNode;
			else if (cursor->parent) 
				cursor->parent->child = newNode;
            cursor->prev = newNode;
			newNode->next = *cursor;
            newNode->initialize(tensor, std::move(args), std::move(name));
			return newNode;
        } else {
            cursor->initialize(tensor, std::move(args), std::move(name));
            cursor.next();
			return cursor->prev;
        }
    }

	void MoveNodeTo(Node* target_place, Node* note_to_move) {
		if (note_to_move->valid()) {
			//remove from current position
			if (note_to_move->parent && note_to_move->parent->child == note_to_move) {
				note_to_move->parent->child = note_to_move->next;
			}
			else if (note_to_move->prev) {
				note_to_move->prev->next = note_to_move->next;
			}
			note_to_move->next->prev = note_to_move->prev;

			//insert into new position
			note_to_move->parent = target_place->parent;
			note_to_move->prev = target_place->prev;
			note_to_move->next = target_place;
			if (target_place->prev) {
				target_place->prev->next = note_to_move;
			}
			else if (target_place->parent) {
				target_place->parent->child = note_to_move;
			}
			target_place->prev = note_to_move;
		}
	}

    void RemoveNode(Node* node) {
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

	[[nodiscard]] map<Node*, Node*> CopyComputation(
	    const unordered_set<Node*>& targets) const;

	void CheckIR(string name, bool check_clustering, bool check_kernels) const;
	void PrintListing(string name, bool compact, map<Node*, string> invalid_nodes) const;
	void GetInputList();
	void GetOutputList();
	void ComputeStatistics();
	void CopyArguments(unordered_set<Arg*> args_to_copy, Node* cursor);
	void ReorderOperations();
	void OptimizeKernels();
	void OptimizeHost();
	void OptimizeOperations();
	void RemoveUnusedOperations();
	void SeparateOperationsIntoKernels();
	void ComputeNodeCost();
	map<Node*, vector<Arg*>> GetKernelOutputs(Node* kernel);
	void AddKernelGlobalMemoryOperations();
	void AddMemoryDeallocation();
	void LinearModeIndices(Tensor*& thread_index, vector<Tensor*>& indices,
	                       Node* cluster, int dims, Tensors kernel_shape);
	void MultiDimensionalModeIndices(Tensor*& thread_index,
	                                 vector<Tensor*>& indices, Node* kernel_,
	                                 int dims, Tensors kernel_shape);
	void FinalizeMemoryIndexing();
	void RemoveUnusedKernels();
	void CompileIR();

	void UpdateGraph() const {
		Node* prev = nullptr;
		for (auto node = begin(); !node.end(); node.next()) {
			node->UpdateEdges();
			node->outputs_.clear();
			if (prev) {
				prev->true_next = *node;
				node->true_prev = prev;
			}
			prev = *node;
		}
		int index = 0;    
		for (auto node = begin(); !node.end(); node.next()) {
			node->UpdateOutputs();
			node->index_ = index++;
		}
	}

	vector<Node*> GetNodesOfType(const string& name) const {
		vector<Node*> result;
		for (auto node = begin(); !node.end(); node.next()) {
			if (node->name == name) {
				result.push_back(*node);
			}
		}
		return result;
	}

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