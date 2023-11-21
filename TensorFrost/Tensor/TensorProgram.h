#pragma once

#include <functional>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Tensor.h"
#include "IR/KernelGen.h"
#include "Backend/Backend.h"

namespace TensorFrost {

using namespace std;

class TensorProgram {
 public:
	using EvaluateFunction = function<Tensors()>;
	EvaluateFunction evaluate_callback;
	IR ir;
	Program* program;
	bool debug = false;

	explicit TensorProgram(EvaluateFunction evaluate)
	    : evaluate_callback(std::move(evaluate)) {
		CreateProgram();
	}

	void CreateProgram();

	vector<TensorMemory*> Evaluate(const vector<TensorMemory*> input);

	~TensorProgram()
	{
		delete program;
	}
};

}  // namespace TensorFrost