#pragma once

#include <functional>
#include <iostream>
#include <list>
#include <unordered_map>
#include <vector>

#include "Operations.h"
#include "Utility/Utility.h"

namespace TensorFrost {
class Tensor;

class Node
{
public:
	Tensor* tensor_;
	bool is_output_;
	int cluster_id_ = -1;

	Node(Tensor* tensor, bool is_output = false)
	    : tensor_(tensor), is_output_(is_output) {}

	//desctructor
	~Node();
};

class IR {
	list<Node> nodes_;

 public:
	void AddNode(Tensor* tensor) { nodes_.emplace_back(tensor); }

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