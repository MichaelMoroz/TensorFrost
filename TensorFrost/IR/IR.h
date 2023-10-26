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
	const Node* node_;
	int index_;

	Argument(Type type, const Node* node, int index)
	    : type_(type), node_(node), index_(index) {}
};

using Arguments = vector<Argument>;
using Tensors = vector<const Tensor*>;

class Node
{
public:
	const string name;
	const Operation* op;
	Tensor* tensor_;
	Arguments arguments_;
	bool is_output_;
	int cluster_id_ = -1;

	Node(Tensor* tensor, Arguments args, string name, bool is_output)
		: tensor_(tensor), is_output_(is_output), arguments_(args), name(name), op(&FindOperation(name)) {
	}
	
	[[nodiscard]] Arguments GetArguments(Argument::Type type) const {
		Arguments result = Arguments();
		for (const auto& input : arguments_) {
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
			result.push_back(argument.node_->tensor_);
		}
		return result;
	}

	~Node();
};

class IR {
	list<Node> nodes_;

 public:
	Node* AddNode(Tensor* tensor, Arguments args, string name) {
		return &nodes_.emplace_back(tensor, args, name, false);
	}

	list<const Node*> GetNodes() const {
		list<const Node*> nodes;
		for (const Node& node : nodes_) {
			nodes.push_back(&node);
		}
		return nodes;
	}

	void Clusterize();
};
}  // namespace TensorFrost