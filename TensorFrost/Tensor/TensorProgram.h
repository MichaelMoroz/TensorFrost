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
	using EvaluateFunction = function<vector<Tensor>()>;
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
		vector<Tensor> outputs = evaluate_callback();
		Tensor::SetEvaluationContext(nullptr);
	}

	static vector<Tensor> Evaluate(const vector<Tensor>& /*inputs*/) {
		return vector<Tensor>();
	}
};

}  // namespace TensorFrost