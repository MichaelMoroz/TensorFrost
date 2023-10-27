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
		CreateExecutionGraph();
	}

	void CreateExecutionGraph() {
		// create new IR graph
		Tensor::SetEvaluationContext(&ir);
		Tensors outputs = evaluate_callback();
		//set outputs
		for (const Tensor* output : outputs) {
			output->SetMemoryType(MemoryType::Output);
		}

		ir.Clusterize();
		ir.PostProcessClusters();
		ir.Clusterize();
		Tensor::SetEvaluationContext(nullptr);
	}

	static Tensors Evaluate(const Tensors& /*inputs*/) { return Tensors(); }

	~TensorProgram() = default;
};

}  // namespace TensorFrost