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
  list<shared_ptr<Tensor>> nodes;

 public:
  void AddNode(shared_ptr<Tensor> node) { nodes.push_back(node); }

  list<shared_ptr<Tensor>> GetNodes() { return nodes; }

  void Clear() { nodes.clear(); }

  string GetOperationListing();
};
}  // namespace TensorFrost