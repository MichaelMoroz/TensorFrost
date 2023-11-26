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

class Argument {
 public:
	enum Type {
		Input,
		Index,
		Shape,
		Memory,
		None,
	};

	Type type_;
	Lable* from_;
	Lable* to_{nullptr};
	int index_;

	Argument(Type type, Lable* node, int index)
	    : type_(type), from_(node), index_(index) {}

	void SetOutput(Lable* output) { to_ = output; }
};

using Arguments = vector<Argument>;
using ArgumentRefs = vector<const Argument*>;
using Tensors = vector<const Tensor*>;

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
	Tensor* tensor_;

	Lable* lable_ = nullptr;

	Lable* cluster_head_ = nullptr;

	Node* prev_ = nullptr;
	Node* next_ = nullptr;

	Arguments inputs_;
	vector<Argument*> outputs_;
	MemoryType memory_type_ = MemoryType::None;
	int memory_index_ = 0;

	Node(Tensor* tensor, Arguments&& args, string&& name)
	    : tensor_(tensor),
	      inputs_(std::move(args)),
	      name(std::move(name)) {
		lable_ = new Lable(this);
		UpdateArgumentOutputs();
		CheckIfValid();
		op = &FindOperation(this->name);
	}

	[[nodiscard]] Lable* GetLable() const { return lable_; }

	void UpdateArgumentOutputs() {
		for (Argument& input : inputs_) {
			input.SetOutput(lable_);
		}
	}

	void SetMemoryType(MemoryType memory_type, int index = 0) {
		memory_type_ = memory_type;
		memory_index_ = index;
	}

	[[nodiscard]] Arguments GetArguments(Argument::Type type) const {
		Arguments result = Arguments();
		for (const auto& input : inputs_) {
			if (input.type_ == type) {
				result.push_back(input);
			}
		}
		// sort by index
		std::sort(result.begin(), result.end(),
		          [](const Argument& a, const Argument& b) {
			          return a.index_ < b.index_;
		          });
		return result;
	}

	[[nodiscard]] map<int, Tensor*> GetArgumentTensors(
	    Argument::Type type) const {
		// get the arguments
		Arguments arguments = GetArguments(type);
		// convert to tensors
		map<int, Tensor*> result = map<int, Tensor*>();
		for (auto& argument : arguments) {
			result[argument.index_] = argument.from_->node_->tensor_;
		}
		return result;
	}

	void RemoveArguments(Argument::Type type) {
		for (auto it = inputs_.begin(); it != inputs_.end();) {
			if (it->type_ == type) {
				it = inputs_.erase(it);
			} else {
				++it;
			}
		}
	}

	void AddArgument(Node* node, Argument::Type type, int index = 0) {
		inputs_.emplace_back(type, node->GetLable(), index);
	}

	void CheckIfValid() const {
		// must have operation
		if (op == nullptr) {
			throw std::runtime_error("Operation not found");
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
	vector<Lable*> cluster_heads;
	map<Lable*, vector<Node*>> output;
	map<Node*, vector<Argument*>> node_output;
	map<Node*, float> node_cost;

	ClusterProp(map<Lable*, vector<Node*>> cluster_outputs,
	            map<Node*, vector<Argument*>> output, map<Node*, float> cost,
	            vector<Lable*> cluster_heads)
	    : output(std::move(cluster_outputs)),
	      node_output(std::move(output)),
	      node_cost(std::move(cost)),
	      cluster_heads(std::move(cluster_heads)) {}
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

		[[nodiscard]] bool is_begin() const { return node_->prev_ == nullptr; }

		[[nodiscard]] bool is_cluster_begin() const {
			return node_ == nullptr || node_->cluster_head_ == nullptr ||
			       node_->cluster_head_->node_ == node_;
		}

		bool is_cluster_end(const Lable* cluster) const {
			return node_ == nullptr || node_->cluster_head_ != cluster;
		}

		Node* get() { return node_; }

		Node* get_next() { return node_->next_; }

		Node* get_prev() { return node_->prev_; }
	};

	Node* AddNode(Tensor* tensor, Arguments&& args, string&& name) {
		Node* new_node = new Node(tensor, std::move(args), std::move(name));
		InsertAfterCursor(new_node);
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
		nodes_.erase(std::remove(nodes_.begin(), nodes_.end(), node), nodes_.end());
		delete node;
	}

	void ExecuteExpressionAfter(Node* node, const function<void()>&& expression,
	                            bool in_cluster = true) {
		// TODO(Moroz): check if no future nodes are used
		Iterator old_cursor = cursor_;
		Lable* old_cluster_head = current_cluster_head_;
		if (!in_cluster) {
			current_cluster_head_ = nullptr;
		} else {
			current_cluster_head_ = node->cluster_head_;
		}
		SetCursor(node);
		expression();
		cursor_ = old_cursor;
		current_cluster_head_ = old_cluster_head;
	}

	void ExecuteExpressionBefore(Node* node, const function<void()>&& expression,
	                             bool in_cluster = true) {
		Iterator old_cursor = cursor_;
		Lable* old_cluster_head = current_cluster_head_;
		if (!in_cluster) {
			current_cluster_head_ = nullptr;
		} else {
			current_cluster_head_ = node->cluster_head_;
		}
		SetCursorBefore(node);
		expression();
		cursor_ = old_cursor;
		current_cluster_head_ = old_cluster_head;
	}

	// reexecute nodes and get map from old to copied nodes
	[[nodiscard]] map<Node*, Node*> CopyComputation(
	    const unordered_set<Node*>& targets) const;

	void OptimizeClusters();

	void RemoveUnusedNodes();

	[[nodiscard]] Iterator begin() const { return begin_; }

	void Clusterize() const;

	void UpdateNodeOutputs() const;

	[[nodiscard]] ClusterProp GetClusterProperties() const;

	void PostProcessClusters();

	void TransformToLinearIndex();

	~IR();

 private:
	vector<Node*> nodes_;
	vector<Node*> cluster_nodes_;
	Iterator cursor_ = Iterator(nullptr);
	Iterator begin_ = Iterator(nullptr);
	Lable* current_cluster_head_ = nullptr;

	void InsertAfterCursor(Node* node) {
		nodes_.push_back(node);
		node->cluster_head_ = current_cluster_head_;
		if (*cursor_ != nullptr) {
			Node* prev_next = cursor_.get_next();
			if (prev_next != nullptr) {
				if (current_cluster_head_ != nullptr &&
				    current_cluster_head_->node_ == prev_next) {
					// if the next node is a cluster head, then we need to update the
					// cluster head
					current_cluster_head_->node_ = node;
				}
				node->next_ = prev_next;
				prev_next->prev_ = node;
			}
			node->prev_ = *cursor_;
			cursor_->next_ = node;
		} else {
			begin_ = Iterator(node);
		}
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
};
}  // namespace TensorFrost