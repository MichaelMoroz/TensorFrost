#include <sstream>
#include <string>
#include "TensorProgram.h"

namespace TensorFrost {

void TensorProgram::CreateProgram() {
	Tensor::SetEvaluationContext(nullptr);

	// create new IR graph
	Tensor::SetEvaluationContext(&ir);
	Tensors outputs = evaluate_callback();
	// set outputs
	for (int i = 0; i < outputs.size(); i++) {
		outputs[i]->SetMemoryType(MemoryType::Output, i);
	}

	program = GenerateProgram(&ir);

	Tensor::SetEvaluationContext(nullptr);

	CompileAndLoadKernel(program);
}

vector<TensorMemory*> TensorProgram::Evaluate(
    const vector<TensorMemory*>& input) const {
	return ExecuteProgram(program, input);
}

string TensorProgram::PrintProperties() const { 
	string properties = "TensorProgram:\n";
	int compute_kernels = program->kernels_.size();
	int lines = 0;
	string line;
	istringstream stream(program->generated_code_);
	while (getline(stream, line)) {
		lines++;
	}
	properties += "  Kernel count: " + to_string(compute_kernels) + "\n";
	properties += "  Intermediate buffers: " + to_string(ir.temp_memory_count) + "\n";
	properties += "  Lines of generated code: " + to_string(lines) + "\n";
	return properties;
}

}  // namespace TensorFrost