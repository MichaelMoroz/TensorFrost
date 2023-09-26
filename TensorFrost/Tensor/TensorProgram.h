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
	using EvaluateFunction = function<Tensors()>;
	EvaluateFunction evaluate_callback;
	IR ir;
	bool debug = false;

	explicit TensorProgram(EvaluateFunction evaluate)
	    : evaluate_callback(std::move(evaluate)) {
		ir = IR();
		CreateExecutionGraph();
	}

	void CreateExecutionGraph() {
		// create new IR graph
		Tensor::SetEvaluationContext(&ir);
		Tensors outputs = evaluate_callback();
		Tensor::SetEvaluationContext(nullptr);
	}

	static Tensors Evaluate(const Tensors& /*inputs*/) { return Tensors(); }
};

}  // namespace TensorFrost