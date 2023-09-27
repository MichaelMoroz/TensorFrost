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

class IR {
	list<Tensor*> nodes_;

 public:
	void AddNode(Tensor* node) { nodes_.push_back(node); }

	list<Tensor*> GetNodes() { return nodes_; }

	void Clear() { nodes_.clear(); }

	string GetOperationListing();

	~IR();
};
}  // namespace TensorFrost