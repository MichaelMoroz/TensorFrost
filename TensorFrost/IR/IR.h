#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Operations.h"
#include "Utility/Utility.h"

namespace TensorFrost {
class Tensor;
class Node;
class Scope;

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
	const string name;
	float cost_ = -1.0f;
	const Operation* op;
	const Tensor* tensor_;

	Lable* lable_ = nullptr;

	Scope* kernel_ = nullptr;
	Scope* scope_ = nullptr;

	vector<Node*> children_;
	pair<Node*, int> parent_ = pair<Node*, int>(nullptr, 0);


	Node* prev_ = nullptr;
	Node* next_ = nullptr;

	Arguments inputs_;
	vector<Arg*> outputs_;
	MemoryType memory_type_ = MemoryType::None;
	int memory_index_ = 0;
	int global_index_ = 0;

	bool has_been_modified_ = false;
	bool is_static = false;

	Node(Tensor* tensor, Arguments&& args, string&& name, bool static_node = false)
	    : tensor_(tensor),
	      inputs_(std::move(args)),
	      name(std::move(name)),
		  is_static(static_node)
	{
		lable_ = new Lable(this);
		UpdateArgumentOutputs();
		op = &FindOperation(this->name);
		CheckClustering();
	}

	int GetDepth() const {
		int depth = 0;
		const Node* current = this;
		while (current->parent_.first != nullptr) {
			depth++;
			current = current->parent_.first;
		}
		return depth;
	}

	Node* NextSibling() const {
		if (next_ != nullptr)
		{
			// just return the next sibling
			return next_;
		}
		else
		{
			if (parent_.first != nullptr) {
				if (parent_.first->children_.size() >
				    parent_.second + 1)  // if there is a next child then go to it
				{
					return parent_.first->children_[parent_.second + 1];
				} else  // if there is no next child then return the parent's next
				        // sibling
				{
					return parent_.first->NextSibling();
				}
			}
			return nullptr;  // no parent, no sibling
		}
		
	}

	Node* Next() const {
		if (children_.empty())
		{
			return NextSibling();
		}	
		return children_[0];
	}

	Node* Prev() const {
		if (prev_ != nullptr)
		{
			return prev_;
		}
		else
		{
			if (parent_.first != nullptr) {
				if (parent_.second > 0)  // if there is a previous child then go to it
				{
					return parent_.first->children_[parent_.second - 1];
				}
				else  // if there is no previous child then return the parent's previous
					 // sibling
				{
					return parent_.first->Prev();
				}
			}
			return nullptr;  // no parent, no sibling
		}
	}

	[[nodiscard]] bool IsLastSibling() const {
		if (parent_.first != nullptr) {
			return parent_.first->children_.size() == parent_.second + 1;
		}
		return false;
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

class Scope {
 public:
	enum ScopeType {
		None,
		Host,
		Kernel,
		HostLoop,
		KernelLoop,
	};

	Node* begin_;
	Lable* shape_node_;
	ScopeType type_ = ScopeType::None;

	Scope(Node* cluster_begin) : begin_(cluster_begin), shape_node_(nullptr) {
		Arguments shape_args = cluster_begin->GetArguments(Arg::Type::Shape);
		if (cluster_begin->name == "memory" || shape_args.size() == 0) {
			type_ = ScopeType::Host;  // allocation and scalar operations
		} else {
			type_ = ScopeType::Kernel;  // multi-dimensional operations
		}
	}
};

class ClusterProp {
 public:
	vector<Scope*> clusters;
	map<Scope*, vector<Node*>> output;
	map<Node*, vector<Arg*>> node_output;

	ClusterProp(map<Scope*, vector<Node*>> cluster_outputs,
	            map<Node*, vector<Arg*>> output,
	            vector<Scope*> clusters)
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

class IR {
 public:
	class Iterator {
	 public:
		enum Type {
			Prev,
			Next,
			Child,
			Parent,
		};

		explicit Iterator(Node* node, Type type = Type::Next, Node* scope = nullptr)
		    : node_(node), type_(type), scope_(scope) {}
		explicit Iterator(const Node* node, Type type = Type::Next,
		                  const Node* scope = nullptr)
		    : node_(const_cast<Node*>(node)), scope_(const_cast<Node*>(scope)), type_(type) {}

		Node* operator*() const { return node_; }

		Node* operator->() { return node_; }

		Iterator& operator++() {
			node_ = get_next();
			return *this;
		}

		Iterator& operator--() {
			node_ = get_prev();
			return *this;
		}

		bool operator!=(const Iterator& other) const {
			return node_ != other.node_;
		}

		bool operator==(const Iterator& other) const {
			return node_ == other.node_;
		}

		[[nodiscard]] bool is_end() const { return node_ == nullptr; }

		[[nodiscard]] bool is_begin() const { return node_ == nullptr; }

		[[nodiscard]] bool is_cluster_begin() const {
			return node_ == nullptr || node_->kernel_ == nullptr ||
			       node_->kernel_->begin_ == node_;
		}

		bool is_cluster_end(const Scope* cluster) const {
			return node_ == nullptr || node_->kernel_ != cluster;
		}

		Node* get() { return node_; }

		Node* get_next() { return node_->Next(); }
		Node* get_next_sibling() { return node_->NextSibling(); }

		Node* get_prev() { return node_->Prev(); }
		Node* get_prev_sibling() { return node_->prev_; }

	private:
		Node* node_ = nullptr;
		Node* scope_ = nullptr;
		Type type_;
	};

	Node* AddNode(Tensor* tensor, Arguments&& args, string&& name) {
		Node* new_node = new Node(tensor, std::move(args), std::move(name));
		InsertAtCursor(new_node);
		return new_node;
	}

	void RemoveNode(Node* node) {
		if (node->prev_ != nullptr) {
			node->prev_->next_ = node->next_;
		}
		if (node->next_ != nullptr) {
			node->next_->prev_ = node->prev_;
		}
		if (node == *cursor_) {
			cursor_ = Iterator(node->prev_);
		}

		if (node->kernel_ != nullptr && node->kernel_->begin_ == node) {
			//assuming the next node is also in the cluster
			node->kernel_->begin_ = node->next_;
		}
		delete node;
	}

	void ExecuteExpressionAfter(Node* node, const function<void()>&& expression,
	                            bool in_cluster = true) {
		// TODO(Moroz): check if no future nodes are used
		Iterator old_cursor = cursor_;
		Scope* old_kernel = current_kernel_;
		Scope* old_scope = current_scope_;
		if (!in_cluster) {
			current_kernel_ = nullptr;
			current_scope_ = nullptr;
		} else {
			current_kernel_ = node->kernel_;
			current_scope_ = node->scope_;
		}
		SetCursor(node);
		expression();
		cursor_ = old_cursor;
		current_kernel_ = old_kernel;
		current_scope_ = old_scope;
	}

	void ExecuteExpressionBefore(Node* node, const function<void()>&& expression,
	                             bool in_cluster = true) {
		Iterator old_cursor = cursor_;
		Scope* old_kernel = current_kernel_;
		Scope* old_scope = current_scope_;
		if (!in_cluster) {
			current_kernel_ = nullptr;
			current_scope_ = nullptr;
		} else {
			current_kernel_= node->kernel_;
			current_scope_ = node->scope_;
		}
		SetCursorBefore(node);
		expression();
		cursor_ = old_cursor;
		current_kernel_ = old_kernel;
		current_scope_ = old_scope;
	}

	void RecomputeGlobalIndices() const;

	void CheckIR(string name, bool check_clustering, bool check_kernels) const;

	// reexecute nodes and get map from old to copied nodes
	[[nodiscard]] map<Node*, Node*> CopyComputation(
	    const unordered_set<Node*>& targets) const;

	void GetInputList();
	void GetOutputList();
	void ComputeStatistics();

	void ReorderOperations();
	void OptimizeKernels();

	void OptimizeOperations();

	void RemoveUnusedOperations();

	[[nodiscard]] Iterator begin() const { return Iterator(begin_node.Next()); }

	void SeparateOperationsIntoKernels() const;

	void PrintListing(string name, bool compact,
	                  map<Node*, string> invalid_nodes) const;

	void UpdateNodeOutputs() const;

	[[nodiscard]] ClusterProp GetClusterProperties() const;

	void AddKernelGlobalMemoryOperations();

	void LinearModeIndices(Tensor*& thread_index, vector<Tensor*>& indices,
	                       Scope* cluster, int dims, Tensors kernel_shape);

	void MultiDimensionalModeIndices(Tensor*& thread_index,
	                                 vector<Tensor*>& indices, Scope* kernel_,
	                                 int dims, Tensors kernel_shape);

	void FinalizeMemoryIndexing();

	void CompileIR();

	void MoveNodeBefore(Node* before, Node* node) {
		if (node == before) {
			return;
		}

		// remove node from current position
		if (node->prev_ != nullptr) {
			node->prev_->next_ = node->next_;
		}
		if (node->next_ != nullptr) {
			node->next_->prev_ = node->prev_;
		}

		// if before is the begin of the cluster, then we need to update the cluster
		// head
		if (before->kernel_ != nullptr && before->kernel_->begin_ == before) {
			before->kernel_->begin_ = node;
		}

		node->next_ = before;
		if (before->prev_ != nullptr)
		{
			before->prev_->next_ = node;
			node->prev_ = before->prev_;
		}
		before->prev_ = node;

		node->kernel_ = before->kernel_;
	}

	void MoveNodeAfter(Node* after, Node* node) {
		if (node == after) {
			return;
		}

		// remove node from current position
		if (node->prev_ != nullptr) {
			node->prev_->next_ = node->next_;
		}
		if (node->next_ != nullptr) {
			node->next_->prev_ = node->prev_;
		}

		node->prev_ = after;
		if (after->next_ != nullptr)
		{
			after->next_->prev_ = node;
			node->next_ = after->next_;
		}
		after->next_ = node;

		node->kernel_ = after->kernel_;
	}

	~IR();

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

	Node begin_node = Node(nullptr, Arguments(), "scope", true);

	vector<Node*> memory_inputs;
	unordered_map<Node*, unordered_map<int, Node*>> shape_memory_map;
	unordered_map<int, Node*> output_memory_map;
	KernelIndexingMode indexing_mode_ = KernelIndexingMode::Linear;
	TensorIndexingMode tensor_indexing_mode_ = TensorIndexingMode::Unsafe;
 private:
	Iterator cursor_ = Iterator(&begin_node, Iterator::Type::Child, &begin_node);
	Scope* current_kernel_ = nullptr;
	Scope* current_scope_ = nullptr;

	void InsertAtCursor(Node* node) {
		node->kernel_ = current_kernel_;
		node->scope_ = current_scope_;
		
			Node* prev_next = cursor_.get_next();
			if (prev_next != nullptr) {
				if (current_kernel_ != nullptr &&
				    current_kernel_->begin_ == prev_next) {
					// if the next node is a cluster head, then we need to update the
					// cluster head
					current_kernel_->begin_ = node;
				}
				if (current_scope_ != nullptr && current_scope_->begin_ == prev_next) {
					// if the next node is a scope head, then we need to update the
					// scope head
					current_scope_->begin_ = node;
				}
				node->next_ = prev_next;
				prev_next->prev_ = node;
			}
			node->prev_ = *cursor_;
			cursor_->next_ = node;
	
		SetCursor(node);
	}

	void SetCursor(Node* node) {
		if (node != nullptr) {
			cursor_ = Iterator(node);
		} else {
			throw std::runtime_error("Cursor cannot be set to nullptr");
		}
	}

	void SetCursorBefore(Node* node) {
		if (node != nullptr) {
			cursor_ = Iterator(node->prev_);
		} else {
			throw std::runtime_error("Node is nullptr");
		}
	}

	void SetCurrentScope(Node* node) {
		current_scope_ = node->scope_;
	}


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