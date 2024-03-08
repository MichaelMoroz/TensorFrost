#pragma once

#include <functional>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

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
	string program_name = "TensorProgram";
	bool debug = false;

	explicit TensorProgram(EvaluateFunction evaluate)
	    : evaluate_callback(std::move(evaluate)) {
		CreateProgram();
	}

	void CreateProgram();

	[[nodiscard]] vector<TensorMemory*> Evaluate(
	    const vector<TensorMemory*>& input) const;

	string PrintProperties() const;

	~TensorProgram() { delete program; }
};

}  // namespace TensorFrost