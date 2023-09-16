#pragma once

#include <functional>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Tensor.h"

namespace TensorFrost {

using namespace std;

class TensorProgram {
 public:
  using EvaluateFunction = function<vector<Tensor>(vector<Tensor>)>;
  EvaluateFunction evaluate_callback;
  IR ir;
  bool debug = false;

  explicit TensorProgram(EvaluateFunction evaluate) : evaluate_callback(std::move(evaluate)) {
    vector<Tensor> inputs = vector<Tensor>();
    ir = IR();
    CreateExecutionGraph(inputs);
  }

  void CreateExecutionGraph(vector<Tensor> inputs) {
    // create new IR graph
    Tensor::SetIR(&ir);
    vector<Tensor> outputs = evaluate_callback(std::move(inputs));
  }

  static vector<Tensor> Evaluate(const vector<Tensor>&  /*inputs*/) { return vector<Tensor>(); }
};

}  // namespace TensorFrost