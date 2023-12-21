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



class Cluster {
 public:
	Node* begin_;
	Lable* shape_node_;

	Cluster(Node* cluster_begin) : begin_(cluster_begin), shape_node_(nullptr) {}
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

class Node {
 public:
	const string name;
	const Operation* op;
	const Tensor* tensor_;

	Lable* lable_ = nullptr;

	Cluster* cluster_ = nullptr;

	Node* prev_ = nullptr;
	Node* next_ = nullptr;

	Arguments inputs_;
	vector<Arg*> outputs_;
	MemoryType memory_type_ = MemoryType::None;
	int memory_index_ = 0;

	bool has_been_modified_ = false;

	Node(Tensor* tensor, Arguments&& args, string&& name)
	    : tensor_(tensor),
	      inputs_(std::move(args)),
	      name(std::move(name)) {
		lable_ = new Lable(this);
		UpdateArgumentOutputs();
		op = &FindOperation(this->name);
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
		if (tensor_ == nullptr) {
			throw std::runtime_error("Tensor not found");
		}
	}

	~Node();
};

void SwapLables(Node* a, Node* b);
void CopyLable(Node* target, Node* copy);

class ClusterProp {
 public:
	vector<Cluster*> clusters;
	map<Cluster*, vector<Node*>> output;
	map<Node*, vector<Arg*>> node_output;
	map<Node*, float> node_cost;

	ClusterProp(map<Cluster*, vector<Node*>> cluster_outputs,
	            map<Node*, vector<Arg*>> output, map<Node*, float> cost,
	            vector<Cluster*> clusters)
	    : output(std::move(cluster_outputs)),
	      node_output(std::move(output)),
	      node_cost(std::move(cost)),
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
		Node* node_ = nullptr;

	 public:
		explicit Iterator(Node* node) : node_(node) {}

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
			return node_ == nullptr || node_->cluster_ == nullptr ||
			       node_->cluster_->begin_ == node_;
		}

		bool is_cluster_end(const Cluster* cluster) const {
			return node_ == nullptr || node_->cluster_ != cluster;
		}

		Node* get() { return node_; }

		Node* get_next() { return node_->next_; }

		Node* get_prev() { return node_->prev_; }
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
		if (node == *begin_) {
			begin_ = Iterator(node->next_);
		}
		if (node->cluster_ != nullptr && node->cluster_->begin_ == node) {
			//assuming the next node is also in the cluster
			node->cluster_->begin_ = node->next_;
		}
		nodes_.erase(std::remove(nodes_.begin(), nodes_.end(), node), nodes_.end());
		delete node;
	}

	void ExecuteExpressionAfter(Node* node, const function<void()>&& expression,
	                            bool in_cluster = true) {
		// TODO(Moroz): check if no future nodes are used
		Iterator old_cursor = cursor_;
		Cluster* old_cluster = current_cluster_;
		if (!in_cluster) {
			current_cluster_ = nullptr;
		} else {
			current_cluster_ = node->cluster_;
		}
		SetCursor(node);
		expression();
		cursor_ = old_cursor;
		current_cluster_ = old_cluster;
	}

	void ExecuteExpressionBefore(Node* node, const function<void()>&& expression,
	                             bool in_cluster = true) {
		Iterator old_cursor = cursor_;
		Cluster* old_cluster_head = current_cluster_;
		if (!in_cluster) {
			current_cluster_ = nullptr;
		} else {
			current_cluster_= node->cluster_;
		}
		SetCursorBefore(node);
		expression();
		cursor_ = old_cursor;
		current_cluster_ = old_cluster_head;
	}

	void CheckIR(string name, bool check_clustering, bool check_kernels) const;

	// reexecute nodes and get map from old to copied nodes
	[[nodiscard]] map<Node*, Node*> CopyComputation(
	    const unordered_set<Node*>& targets) const;

	void OptimizeKernels();

	void OptimizeOperations();

	void RemoveUnusedOperations();

	[[nodiscard]] Iterator begin() const { return begin_; }

	[[nodiscard]] Iterator end() const { return end_; }

	void SeparateOperationsIntoKernels() const;

	void PrintListing(string name, bool compact,
	                  map<Node*, string> invalid_nodes) const;

	void UpdateNodeOutputs() const;

	[[nodiscard]] ClusterProp GetClusterProperties() const;

	void AddKernelGlobalMemoryOperations();

	void LinearModeIndices(Tensor*& thread_index, vector<Tensor*>& indices,
	                       Cluster* cluster, int dims, Tensors kernel_shape);

	void MultiDimensionalModeIndices(Tensor*& thread_index,
	                                 vector<Tensor*>& indices, Cluster* cluster_,
	                                 int dims, Tensors kernel_shape);

	void FinalizeMemoryIndexing();

	void CompileIR();

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

	vector<Node*> nodes_;
	KernelIndexingMode indexing_mode_ = KernelIndexingMode::Linear;
	TensorIndexingMode tensor_indexing_mode_ = TensorIndexingMode::Unsafe;
 private:
	vector<Node*> cluster_nodes_;
	Iterator cursor_ = Iterator(nullptr);
	Iterator cursor_next_ = Iterator(nullptr);
	Iterator begin_ = Iterator(nullptr);
	Iterator end_ = Iterator(nullptr);
	Cluster* current_cluster_ = nullptr;

	void InsertAtCursor(Node* node) {
		nodes_.push_back(node);
		node->cluster_ = current_cluster_;
		if (*cursor_ != nullptr) {
			Node* prev_next = cursor_.get_next();
			if (prev_next != nullptr) {
				if (current_cluster_ != nullptr &&
				    current_cluster_->begin_ == prev_next) {
					// if the next node is a cluster head, then we need to update the
					// cluster head
					current_cluster_->begin_ = node;
				}
				node->next_ = prev_next;
				prev_next->prev_ = node;
			}
			node->prev_ = *cursor_;
			cursor_->next_ = node;
		} 
		else
		{
			if (*cursor_next_ != nullptr) {
				cursor_next_->prev_ = node;
				node->next_ = *cursor_next_;
			}
			begin_ = Iterator(node);
		}
		if (node->next_ == nullptr) {
			end_ = Iterator(node);
		}
		SetCursor(node);
	}

	void SetCursor(Node* node) {
		if (node != nullptr) {
			cursor_ = Iterator(node);
			cursor_next_ = Iterator(node->next_);
		} else {
			throw std::runtime_error("Cursor cannot be set to nullptr");
		}
	}

	void SetCursorBefore(Node* node) {
		if (node != nullptr) {
			cursor_ = Iterator(node->prev_);
			cursor_next_ = Iterator(node);
		} else {
			throw std::runtime_error("Node is nullptr");
		}
	}
};

bool CompareShape(const Node* a, const Node* b);

}  // namespace TensorFrost