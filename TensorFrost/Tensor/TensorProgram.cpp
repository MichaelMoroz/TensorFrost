#include <sstream>
#include <string>
#include "TensorProgram.h"

namespace TensorFrost {

void TensorProgram::CreateProgram(string name) {
	Tensor::SetEvaluationContext(nullptr);

	//get current time
	auto start = std::chrono::high_resolution_clock::now();

	// create new IR graph
	Tensor::SetEvaluationContext(&ir);
	Tensors outputs = evaluate_callback();
	// set outputs
	for (int i = 0; i < outputs.size(); i++) {
		outputs[i]->SetMemoryType(MemoryType::Output, i);
	}

	if (outputs.size() == 0) {
		throw std::runtime_error("TensorProgram does not do any computation: no outputs");
	}

	program = GenerateProgram(&ir);
	program->program_name = name;

	Tensor::SetEvaluationContext(nullptr);

	GenerateCode(program);

	//get current time
	auto end = std::chrono::high_resolution_clock::now();
	compile_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0f;

	CompileAndLoadKernelModule(program);

	auto external_end = std::chrono::high_resolution_clock::now();
	external_compile_time = std::chrono::duration_cast<std::chrono::nanoseconds>(external_end - end).count() / 1000000.0f;
}

vector<TensorMemory*> TensorProgram::Evaluate(
    const vector<TensorMemory*>& input) const {
	return ExecuteProgram(program, input);
}

string TensorProgram::PrintProperties() const { 
	string properties = program->program_name + ":\n";
	int compute_kernels = (int)program->kernels_.size();
	int lines = 0;
	string line;
	istringstream stream(program->generated_code_);
	while (getline(stream, line)) {
		lines++;
	}
	properties += "  Kernel count: " + to_string(compute_kernels) + "\n";
	properties += "  Intermediate buffers: " + to_string(ir.temp_memory_count) + "\n";
	properties += "  Lines of generated code: " + to_string(lines) + "\n";
	properties += "  IR Compile time: " + to_string(compile_time) + " ms\n";
	properties += "  Compiler time: " + to_string(external_compile_time) + " ms\n";
	return properties;
}

}  // namespace TensorFrost