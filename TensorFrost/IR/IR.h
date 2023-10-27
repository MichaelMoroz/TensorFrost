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
		RefCopy,
		Loop,
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

static void SwapLables(Node* a, Node* b) 
{
	// first swap the node addresses
	a->lable_->node_ = b;
	b->lable_->node_ = a;

	// now swap the labels
	NodeLable* temp = a->lable_;
	a->lable_ = b->lable_;
	b->lable_ = temp;
}

class IR {
	vector<Node*> nodes_;

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

		bool is_end() const {
			return node_ == nullptr;
		}

		bool is_begin() const {
			return node_->prev_ == nullptr;
		}

		Node* get()  {
			return node_;
		}

		Node* get_next() {
			return node_->next_;
		}

		Node* get_prev() {
			return node_->prev_;
		}
	};
	iterator cursor_ = iterator(nullptr);
	iterator begin_ = iterator(nullptr);

	Node* AddNode(Tensor* tensor, Arguments args, string name) {
		Node* new_node = new Node(tensor, args, name);
		InsertAfterCursor(new_node);
		return new_node;
	}

	void InsertAfterCursor(Node* node)
	{
		nodes_.push_back(node);
		if (*cursor_ != nullptr) {
			Node* prev_next = cursor_.get_next();
			if (prev_next != nullptr) {
				node->next_ = prev_next;
				prev_next->prev_ = node;
			}
			node->prev_ = *cursor_;
			cursor_->next_ = node;
		}
		else
		{
			begin_ = iterator(node);
		}
		SetCursor(node);
	}

	void SetCursor(Node* node) {
		if (node != nullptr) 
		{
			cursor_ = iterator(node);
		}
		else
		{
			throw std::runtime_error("Cursor cannot be set to nullptr");
		}
	}

	iterator begin() const {
		return begin_;
	}

	void Clusterize();

	void UpdateNodeOutputs();

	void PostProcessClusters();

	~IR();
};
}  // namespace TensorFrost