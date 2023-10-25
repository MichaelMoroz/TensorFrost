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

class Argument {
 public:
	enum Type {
		Input,
		Index,
		Shape,
		RefCopy,
		Loop,
	};

	Type type_;
	const Tensor* tensor_;
	int index_;

	Argument(Type type, const Tensor* tensor, int index)
	    : type_(type), tensor_(tensor), index_(index) {}
};

using Arguments = vector<Argument>;
using Tensors = vector<const Tensor*>;

class Node
{
public:
	Tensor* tensor_;
	Arguments arguments_;
	bool is_output_;
	int cluster_id_ = -1;

	Node(Tensor* tensor, Arguments args, bool is_output)
	    : tensor_(tensor), is_output_(is_output), arguments_(args) {}

	
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
			result.push_back(argument.tensor_);
		}
		return result;
	}
	[[nodiscard]] int GetDimension() const {
		// find max dimension
		int max_dim = -1;

		for (const auto& input : arguments_) {
			if (input.type_ == Argument::Type::Shape) {
				max_dim = std::max(max_dim, input.index_);
			}
		}

		return max_dim + 1;
	}

	[[nodiscard]] vector<const Tensor*> GetShape() const {
		vector<const Tensor*> result = vector<const Tensor*>();
		// get max dimension
		int max_dim = -1;
		for (const auto& input : arguments_) {
			if (input.type_ == Argument::Type::Shape) {
				max_dim = std::max(max_dim, input.index_);
			}
		}

		if (max_dim == -1) {
			return result;
		}

		// resize result
		result.resize(max_dim + 1);
		for (int i = 0; i <= max_dim; i++) {
			result[i] = nullptr;
		}
		// fill result
		for (const auto& input : arguments_) {
			if (input.type_ == Argument::Type::Shape) {
				result[input.index_] = input.tensor_;
			}
		}
		// if there are any missing dimensions, fill them with 1
		//Tensor& one = Constant(1);
		//for (int i = 0; i <= max_dim; i++) {
		//	if (result[i] == nullptr) {
		//		result[i] = &one;
		//	}
		//}
		return result;
	}
	[[nodiscard]] vector<int> TryGetShape() const {
		vector<int> result = vector<int>();
		// get max dimension
		int max_dim = -1;
		for (const auto& input : arguments_) {
			if (input.type_ == Argument::Type::Shape) {
				max_dim = std::max(max_dim, input.index_);
			}
		}

		if (max_dim == -1) {
			return result;
		}

		// resize result
		result.resize(max_dim + 1);
		for (int i = 0; i <= max_dim; i++) {
			result[i] = 1;
		}
		// fill result
		for (const auto& input : arguments_) {
			if (input.type_ == Argument::Type::Shape) {
				result[input.index_] = AsInt(input.tensor_->data[0]);
			}
		}
		return result;
	}

	//desctructor
	~Node();
};

class IR {
	list<Node> nodes_;

 public:
	void AddNode(Tensor* tensor, Arguments args) { nodes_.emplace_back(tensor, args, false); }

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