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
	int intermediate_buffers = 0;
	int compute_kernels = 0;
	for (int i = 0; i < program->kernels_.size(); i++) {
		if (program->kernels_[i].type_ == KernelType::Host) {
			intermediate_buffers++;
		}
		else {
			compute_kernels++;
		}
	}
	int lines = 0;
	string line;
	istringstream stream(program->generated_code_);
	while (getline(stream, line)) {
		lines++;
	}
	properties += "  Kernel count: " + to_string(compute_kernels) + "\n";
	properties += "  Intermediate buffers: " + to_string(intermediate_buffers) + "\n";
	properties += "  Lines of generated code: " + to_string(lines) + "\n";
	properties += "  IR size: " + to_string(ir.nodes_.size()) + "\n";
	return properties;
}

}  // namespace TensorFrost