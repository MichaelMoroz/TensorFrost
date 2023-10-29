#pragma once

#include <functional>
#include <iostream>
#include <list>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include "Operations.h"
#include "Utility/Utility.h"

namespace TensorFrost {
class Tensor;
class Node;

class NodeLable {
 public:
	Node* node_;
	NodeLable(Node* node) : node_(node) {}

	Node* operator*() const { return node_; }
	Node* operator->() { return node_; }
	Node* get() { return node_; }
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
	NodeLable* from_;
	NodeLable* to_;
	int index_;

	Argument(Type type, NodeLable* node, int index)
	    : type_(type), from_(node), index_(index), to_(nullptr) {}

	void SetOutput(NodeLable* output) {
		to_ = output;
	}
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

class Node
{
public:
	NodeLable* lable_ = nullptr;

	Node* prev_ = nullptr;
	Node* next_ = nullptr;
	
	const string name;
	const Operation* op;
	Tensor* tensor_;
	Arguments inputs_;
	vector<const Argument*> outputs_;
	MemoryType memory_type_ = MemoryType::None;
	int cluster_id_ = -1;

	Node(Tensor* tensor, Arguments args, string name)
	    : tensor_(tensor),
	      inputs_(args),
	      name(name),
	      op(&FindOperation(name)) 
	{
		lable_ = new NodeLable(this);
		UpdateArgumentOutputs();
	}

	NodeLable* GetLable() {
		return lable_;
	}

	void UpdateArgumentOutputs() {
		for (Argument& input : inputs_) {
			input.SetOutput(lable_);
		}
	}

	void SetMemoryType(MemoryType memory_type) {
		memory_type_ = memory_type;
	}
	
	[[nodiscard]] Arguments GetArguments(Argument::Type type) const {
		Arguments result = Arguments();
		for (const auto& input : inputs_) {
			if (input.type_ == type) {
				result.push_back(input);
			}
		}
		// sort by index
		std::sort(
		    result.begin(), result.end(),
		          [](const Argument& a, const Argument& b) {
			          return a.index_ < b.index_;
		          });
		return result;
	}

	[[nodiscard]] Tensors GetArgumentTensors(Argument::Type type) const {
		// get the arguments
		Arguments arguments = GetArguments(type);
		// convert to tensors
		Tensors result = Tensors();
		for (const auto& argument : arguments) {
			result.push_back(argument.from_->get()->tensor_);
		}
		return result;
	}

	~Node();
};

void SwapLables(Node* a, Node* b);
void CopyLable(Node* target, Node* copy);

class ClusterProp
{
public:
	map<int, vector<Node*>> output;
	map<int, Node*> begin;
	map<int, Node*> end;

	ClusterProp(map<int, vector<Node*>> cluster_outputs, map<int, Node*> cluster_begin, map<int, Node*> cluster_end)
		: output(cluster_outputs), begin(cluster_begin), end(cluster_end) {}
};

class IR {
public:
	class iterator {
		Node* node_ = nullptr;

	 public:
		iterator(Node* node) : node_(node) {}

		Node* operator*() const { return node_; }

		Node* operator->() { return node_; }

		iterator& operator++() {
			node_ = get_next();
			return *this;
		}

		iterator& operator--() {
			node_ = get_prev();
			return *this;
		}

		bool operator!=(const iterator& other) const {
			return node_ != other.node_;
		}

		bool operator==(const iterator& other) const {
			return node_ == other.node_;
		}

		bool is_end() const { return node_ == nullptr; }

		bool is_begin() const { return node_->prev_ == nullptr; }

		bool is_cluster_begin() const { return node_->prev_ == nullptr || node_->prev_->cluster_id_ != node_->cluster_id_; }

		bool is_cluster_end() const { return node_->next_ == nullptr || node_->next_->cluster_id_ != node_->cluster_id_; }

		Node* get() { return node_; }

		Node* get_next() { return node_->next_; }

		Node* get_prev() { return node_->prev_; }
	};

	Node* AddNode(Tensor* tensor, Arguments args, string name) {
		Node* new_node = new Node(tensor, args, name);
		InsertAfterCursor(new_node);
		return new_node;
	}

	void ExecuteExpressionAfter(Node* node, function<void()> expression) {
		//TODO check if no future nodes are used
		iterator old_cursor = cursor_;
		int old_cluster_id = current_cluster_id_;
		current_cluster_id_ = node->cluster_id_;
		SetCursor(node);
		expression();
		cursor_ = old_cursor;
		current_cluster_id_ = old_cluster_id;
	}

	void ExecuteExpressionBefore(Node* node, function<void()> expression) {
		iterator old_cursor = cursor_;
		int old_cluster_id = current_cluster_id_;
		current_cluster_id_ = node->cluster_id_;
		SetCursorBefore(node);
		expression();
		cursor_ = old_cursor;
		current_cluster_id_ = old_cluster_id;
	}

	iterator begin() const { return begin_; }

	void Clusterize();

	void UpdateNodeOutputs();

	ClusterProp GetClusterProperties();

	void PostProcessClusters();

	void TransformToLinearIndex();

	~IR();

 private:
	vector<Node*> nodes_;
	iterator cursor_ = iterator(nullptr);
	iterator begin_ = iterator(nullptr);
	int current_cluster_id_ = -1;

	void InsertAfterCursor(Node* node) {
		nodes_.push_back(node);
		if (*cursor_ != nullptr) {
			Node* prev_next = cursor_.get_next();
			if (prev_next != nullptr) {
				node->next_ = prev_next;
				prev_next->prev_ = node;
			}
			node->prev_ = *cursor_;
			cursor_->next_ = node;
		} else {
			begin_ = iterator(node);
		}
		node->cluster_id_ = current_cluster_id_;
		SetCursor(node);
	}

	void SetCursor(Node* node) {
		if (node != nullptr) {
			cursor_ = iterator(node);
		} else {
			throw std::runtime_error("Cursor cannot be set to nullptr");
		}
	}

	void SetCursorBefore(Node* node) {
		if (node != nullptr) {
			cursor_ = iterator(node->prev_);
		} else {
			throw std::runtime_error("Node is nullptr");
		}
	}
};
}  // namespace TensorFrost