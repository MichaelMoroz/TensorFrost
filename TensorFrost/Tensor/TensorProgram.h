#pragma once

#include <functional>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <chrono>

#include "Backend/Backend.h"
#include "Compiler/KernelGen.h"
#include "Tensor.h"

namespace TensorFrost {

using namespace std;

class TensorProgram {
 public:
	static size_t program_id;
	using EvaluateFunction = function<Tensors()>;
	EvaluateFunction evaluate_callback;
	IR ir;
	Program* program;
	bool debug = false;
	float compile_time = 0.0f;
	float codegen_time = 0.0f;
	float host_compile_time = 0.0f;
	float shader_compile_time = 0.0f;

	explicit TensorProgram(EvaluateFunction evaluate, string name) : evaluate_callback(std::move(evaluate)) {
		CreateProgram(name);
		program_id++;
	}

	void CreateProgram(string name);

	vector<TFTensor*> Evaluate(
	    const vector<TFTensor*>& input) const;

	string PrintProperties() const;

	~TensorProgram() { delete program; }
};

}  // namespace TensorFrost