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
	Node* get() const { return node_; }
};

enum class ArgType {
	Input,
	Index,
	Shape,
	Memory,
	None,
	Count,
};

class Arg {
 public:
	static inline const map<ArgType, string> type_names = {
	    {ArgType::Input, "Input"}, {ArgType::Index, "Index"}, {ArgType::Shape, "Shape"},
		{ArgType::Memory, "Memory"}, {ArgType::None, "None"},
	};

	static string TypeToString(ArgType type) { return type_names.at(type); }

	ArgType type_;
	Lable* from_;
	Lable* to_{nullptr};
	int index_;

	Arg(ArgType type, Lable* node, int index)
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
	Constant,
};

enum class TensorIndexingMode {
	Unsafe,
	Clamp,
	Repeat,
	Zero,
};


using ArgID = pair<ArgType, int>;

struct HashArgID {
	size_t operator()(const ArgID& id) const {
		return (int)id.first + id.second * (int)ArgType::Count;
	}
};

class ArgumentManager {
private:
	bool add_parenthesis = false;
	unordered_map<ArgID, DataType, HashArgID> argument_types_;
	unordered_map<ArgType, int> argument_counts_;
	unordered_map<ArgID, string, HashArgID> argument_names_;
	unordered_map<ArgID, bool, HashArgID> argument_requires_parenthesis_;

public:
	unordered_map<ArgID, Node*, HashArgID> arguments_;

	ArgumentManager() {}

	void AddParenthesis(bool add) {
		add_parenthesis = add;
	}

	void AddArgument(Arg* arg);

	void SetName(ArgID id, string name, bool requires_parenthesis = false) {																																																		
		argument_names_[id] = name; 
	    argument_requires_parenthesis_[id] = requires_parenthesis;
	}

	bool Has(ArgType type, int index = 0) {
		ArgID id = ArgID(type, index);
		return arguments_.find(id) != arguments_.end();
	}

	Node* Get(ArgType type, int index = 0) {
		ArgID id = ArgID(type, index);
		auto Arg = arguments_.find(id);
		if (Arg != arguments_.end()) {
			return Arg->second;
		} else {
			throw std::runtime_error("Argument not found");
		}
	}

	DataType Type(ArgType type, int index = 0) {
		ArgID id = ArgID(type, index);
		auto Arg = argument_types_.find(id);
		if (Arg != argument_types_.end()) {
			return Arg->second;
		}
		else {
			throw std::runtime_error("Argument type not found");
		}
	}

	int Count(ArgType type) {
		auto Arg = argument_counts_.find(type);
		if (Arg != argument_counts_.end()) {
			return Arg->second;
		}
		else {
			return 0;
		}
	}

	string Name(ArgType type, int index = 0) {
		ArgID id = ArgID(type, index);
		auto Arg = argument_names_.find(id);
		if (Arg != argument_names_.end()) {
			string name = Arg->second;
			if (add_parenthesis && argument_requires_parenthesis_[id]) {
				name = "(" + name + ")";
			}
			return name;
		}
		else {
			throw std::runtime_error("Argument name not found");
		}
	}
};

class Node {
 public:
	int index_ = -1;
	string var_name = "none";
	string debug_name;
	string name;
	float cost_ = -1.0f;
	
	Node *parent = nullptr, *child = nullptr, *next = nullptr, *prev = nullptr;
    bool placeholder = true;

	//only true after graph has been updated
	Node *true_prev = nullptr, *true_next = nullptr;
	
	const Operation* op;
	const Tensor* tensor_;

	Lable* lable_ = nullptr;

	Arguments inputs_;
	vector<Arg*> outputs_;
	MemoryType memory_type_ = MemoryType::None;
	int special_index_ = 0;

	bool has_been_modified_ = false;
	bool is_static = false;

	Node(Node* prev = nullptr, Node* parent = nullptr) : parent(parent), prev(prev) {}

    bool valid() {
        return !placeholder;
    }

	//clamp unless otherwise specified
	TensorIndexingMode indexing_mode_;

	//kernel properties
	vector<int> group_size;

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
		op = FindOperation(name);
		CheckNode();
		indexing_mode_ = TensorIndexingMode::Clamp;
    }

	void CopyProperties(Node* other) {
		debug_name = other->debug_name;
		name = other->name;
		special_index_ = other->special_index_;
		has_been_modified_ = other->has_been_modified_;
		is_static = other->is_static;
		indexing_mode_ = other->indexing_mode_;
		group_size = other->group_size;
	}

	const Tensor* GetTensor() const;

	Lable* GetLable() const { return lable_; }

	void SetAsModified()
	{
		has_been_modified_ = true;
	}

	bool HasBeenModified() const
	{
		return has_been_modified_;
	}

	int TryComputeShape();

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

	void MakeOutputsUseGivenNode(Node* replacement) {
		for (Arg* output : outputs_) {
			output->from_ = replacement->GetLable();
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

	bool HasParent(string name) {
		return GetParent(name) != this;
	}

	void SetMemoryType(MemoryType memory_type, int index = 0) {
		if (memory_type_ != MemoryType::None) {
			throw std::runtime_error("Memory type already set. Are you trying to output an input?");
		}
		memory_type_ = memory_type;
		special_index_ = index;
	}

	int MaxIndex(ArgType type) const {
		int max_index = 0;
		for (const auto& input : inputs_) {
			if (input.type_ == type) {
				max_index = std::max(max_index, input.index_);
			}
		}
		return max_index;
	}

	Arguments GetArguments(ArgType type) const {
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

	ArgMap GetArgumentMap(ArgType type) const {
		ArgMap result = ArgMap();
		for (auto& input : inputs_) {
			if (input.type_ == type) {
				result[input.index_] = &input;
			}
		}
		return result;
	}
	
	ArgumentManager GetArgumentManager() {
		ArgumentManager result = ArgumentManager();
		for (auto& input : inputs_) {
			result.AddArgument(&input);
		}
		return result;
	}

	map<int, const Tensor*> GetArgumentTensors(
	    ArgType type) const {
		// get the arguments
		Arguments arguments = GetArguments(type);
		// convert to tensors
		map<int, const Tensor*> result = map<int, const Tensor*>();
		for (auto& argument : arguments) {
			result[argument.index_] = argument.from_->node_->GetTensor();
		}
		return result;
	}

	void RemoveArguments(ArgType type) {
		for (auto it = inputs_.begin(); it != inputs_.end();) {
			if (it->type_ == type) {
				it = inputs_.erase(it);
			} else {
				++it;
			}
		}
	}

	void AddArgument(Node* node, ArgType type, int index = 0) {
		inputs_.emplace_back(type, node->GetLable(), index);
	}

	void CheckNode() const {
		// must have operation
		if (op == nullptr) {
			throw std::runtime_error("Operation object not found");
		}

		// must have tensor
		if (tensor_ == nullptr && !is_static) {
			throw std::runtime_error("Tensor not found");
		}
	}

	Node* GetLastVersion(Node* latest_node) {
		//find last store/scatter operation
		Node* last_modifier = this;
		int last_index = -1;
		Node* loop_node = latest_node->GetParent("loop");
		bool has_loop = loop_node != latest_node;
		for (auto& output : outputs_) {
			if (output->type_ != ArgType::Memory) {
				continue;
			}
			Node* output_node = output->to_->get();
			if (output_node->op->HasAllTypes(OpType::Modifier)) {
				if (output_node->index_>last_index) {
					// either find the last modifier or the last memory node
					// or if there is a loop, find the last modifier inside the loop (i.e.
					// the previous iteration's modifier)
					// if the loop is scalar, then it doesn't matter
					bool before_latest = output_node->index_ < latest_node->index_;
					bool inside_loop = has_loop && output_node->HasParent(loop_node);
					bool not_same = output_node != latest_node;
					if ((before_latest || inside_loop) && not_same)
					{
						last_index = output_node->index_;
						last_modifier = output_node;
					}
				}
			}
		}
		return last_modifier;
	}

	Node* GetFinalVersion() {
		Node* final_version = this;
		int last_index = -1;
		for (auto& output : outputs_) {
			if (output->type_ != ArgType::Memory) {
				continue;
			}
			Node* output_node = output->to_->get();
			if (output_node->op->HasAllTypes(OpType::Modifier) && !output_node->op->HasAllTypes(OpType::MemoryOp)) {
				if (output_node->index_ > last_index) {
					last_index = output_node->index_;
					final_version = output_node;
				}
			}
		}
		return final_version;
	}

	~Node();
};

void SwapLables(Node* a, Node* b);
void CopyLable(Node* target, Node* copy);

//NodeIterator is a depth first iterator that iterates through the child nodes of a root node
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

	
	NodeIterator& go_to_next() {
		if (!currentNode) {
			throw std::runtime_error("Invalid node");
		}

		currentNode = currentNode->next;

		return *this;
	}

	NodeIterator& go_to_parent() {
		if (!currentNode) {
			throw std::runtime_error("Invalid node");
		}

		currentNode = currentNode->parent;

		return *this;
	}

	NodeIterator& go_to_child() {
		if (!currentNode) {
			throw std::runtime_error("Invalid node");
		}

		currentNode = currentNode->child;

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

class ShapeInfo {
 public:
	map<int, Node*> shape;
	int dim = 0;
	string name;

	ShapeInfo() {}

	ShapeInfo(ShapeInfo* shape_info) {
		shape = shape_info->shape;
		dim = shape_info->dim;
		name = shape_info->name;
	}

	ShapeInfo(ArgMap shape, string name) {
		for (auto& [index, arg] : shape) {
			AddShape(arg->index_, arg->from_->node_);
		}
		this->name = name;
	}

	ShapeInfo(const Node* node)
	    : ShapeInfo(node->GetArgumentMap(ArgType::Shape),
	                node->var_name != "" ? node->var_name : node->name) {}

	void AddShape(int index, Node* node) {
		shape[index] = node;
		dim = max(dim, index + 1);
	}

	Tensors GetTensors() const {
		Tensors tensors = Tensors();
		tensors.resize(dim);
		for (auto& [index, node] : shape) {
			tensors[index] = node->GetTensor();
		}
		return tensors;
	}
};

struct ShapeCompareResult {
	bool compatible;
	bool is_broadcast;
	ShapeInfo broadcast_shape;
	int broadcast_dim;
	int a_dim;
	int b_dim;
	int min_dim;
};

ShapeCompareResult CompareShape(const Node* a, const Node* b,
                                bool exact_match = false,
                                bool throw_error = false);

ShapeCompareResult CompareShape(ShapeInfo& a, ShapeInfo& b,
                                bool exact_match = false,
                                bool throw_error = false);


class KernelScope {
 public:
	Node* begin = nullptr;
	Node* end = nullptr;
	ShapeInfo scope_shape;
	unordered_set<Node*> boundary_nodes;

	static bool IsBoundary(const Node* input, const Node* output, ArgType arg_type, bool is_identity);

	KernelScope() : begin(nullptr), end(nullptr) {}
	KernelScope(Node* node, unordered_set<KernelScope*>& output_scopes);

	KernelScope(Node* begin, Node* end, ShapeInfo shape, unordered_set<Node*> boundary_nodes)
	    : begin(begin), end(end), scope_shape(shape), boundary_nodes(boundary_nodes) {}
	
	void CopyProperties(KernelScope* other) {
		begin = other->begin;
		end = other->end;
		scope_shape = other->scope_shape;
		boundary_nodes = other->boundary_nodes;
	}

	bool IsValid() const;

	static std::unordered_set<KernelScope*> ComputeScopes(Node* root);

	static KernelScope* Merge(KernelScope* a, KernelScope* b);

	void AddBoundaryNodes(unordered_set<Node*> new_boundary_nodes) {
		boundary_nodes.insert(new_boundary_nodes.begin(), new_boundary_nodes.end());
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
			cursor.go_to_next();
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

	stack<Node*> scope_stack;

	void BeginScope(Node* node) {
		scope_stack.push(*cursor);
		SetCursor(node);
	}

	void EndScope() {
		if (scope_stack.empty()) throw std::runtime_error("No scope to end");
		SetCursor(scope_stack.top());
		scope_stack.pop();
	}

    void ExecuteExpressionAfter(Node* node, const function<void()>&& expression) {
		BeginScope(node->next);
		expression();
		EndScope();
    }

    void ExecuteExpressionBefore(Node* node, const function<void()>&& expression) {
		BeginScope(node);
		expression();
		EndScope();
    }

	void ExecuteExpressionChild(Node* node, const function<void()>&& expression) {
		BeginScope(node->child);
		expression();
		EndScope();
	}

	void CheckIR(string name, bool check_clustering, bool check_kernels) const;
	string PrintListing(map<Node*, string> node_debug) const;
	map<Node*, Node*> CopyComputation(const unordered_set<Node*>& targets,
	                                  const unordered_map<int, Node*>& indices);
	void GetInputList();
	void GetOutputList();
	void ComputeStatistics();
	void CopyArguments(unordered_set<Arg*> args_to_copy, Node* cursor);
	map<Node*, Node*> CopyNodesWithIndex(unordered_set<Node*> nodes_to_copy,
	                          unordered_map<int, Node*> indices, Node* cursor);
	void ReorderOperations();
	void MoveShapeOutsideKernels();
	void OptimizeKernels();
	void OptimizeHost();
	void OptimizeOperations();
	void OptimizeKernelLoadOperations();
	void RemoveUnusedOperations();
	void InsertAlgorithmicPrimitives();
	void SeparateOperationsIntoKernels();
	void ComputeNodeCost();
	map<Node*, vector<Arg*>> GetKernelOutputs(Node* kernel);
	void AddNodeLoadOperations(Node* node, Node* kernel, Tensors indices);
	void AddKernelGlobalLoadOperations();
	void AddMemoryOpIndices();
	void AddKernelGlobalStoreOperations();
	void CheckKernelShapes();
	void AddMemoryDeallocation();
	void ReplaceDimNodes(Node* kernel, vector<Tensor*> indices, int dims);
	void LinearModeIndices(vector<Tensor*>& indices, Node* kernel, int dims,
	                       Tensors kernel_shape);
	void MultiDimensionalModeIndices(vector<Tensor*>& indices, Node* kernel_,
	                                 int dims, Tensors kernel_shape);
	Tensor* LinearBlockModeIndices(vector<Tensor*>& indices, Node* kernel_, int dims,
	                            Tensors kernel_shape);
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

	vector<Node*> GetNodesOfType(OpType type) const {
		vector<Node*> result;
		for (auto node = begin(); !node.end(); node.next()) {
			if (node->op->HasAllTypes(type)) {
				result.push_back(*node);
			}
		}
		return result;
	}

	int input_memory_count = 0;
	int output_memory_count = 0;
	int temp_memory_count = 0;

	int readbacks = 0;
	int writebacks = 0;

	vector<Node*> memory_inputs;
	unordered_map<Node*, unordered_map<int, Node*>> shape_memory_map;
	unordered_map<int, Node*> output_memory_map;
};
}  // namespace TensorFrost