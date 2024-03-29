#pragma once

#include <functional>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <chrono>

#include "Backend/Backend.h"
#include "IR/KernelGen.h"
#include "Tensor.h"

namespace TensorFrost {

using namespace std;

class TensorProgram {
 public:
	using EvaluateFunction = function<Tensors()>;
	EvaluateFunction evaluate_callback;
	IR ir;
	Program* program;
	bool debug = false;
	float compile_time = 0.0f;
	float external_compile_time = 0.0f;

	explicit TensorProgram(EvaluateFunction evaluate, string name) : evaluate_callback(std::move(evaluate)) {
		CreateProgram(name);
	}

	void CreateProgram(string name);

	vector<TensorMemory*> Evaluate(
	    const vector<TensorMemory*>& input) const;

	string PrintProperties() const;

	~TensorProgram() { delete program; }
};

}  // namespace TensorFrost