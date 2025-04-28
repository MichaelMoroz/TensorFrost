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
	Tensor::BeginRegion(name);
	Tensors outputs = evaluate_callback();
	Tensor::EndRegion(name);
	// set outputs
	for (int i = 0; i < outputs.size(); i++) {
		outputs[i]->SetMemoryType(NodeProp::OutputMemory, i);
	}
	ir.output_memory_count = (int)outputs.size();

	if (outputs.size() == 0) {
		throw std::runtime_error("TensorProgram does not do any computation: no outputs");
	}

	program = GenerateProgram(&ir);
	program->program_name = name;

	Tensor::SetEvaluationContext(nullptr);

	auto end = std::chrono::high_resolution_clock::now();

	compile_time = std::chrono::duration<float, std::milli>(end - start).count();

	start = std::chrono::high_resolution_clock::now();

	GenerateCode(program);

	end = std::chrono::high_resolution_clock::now();

	codegen_time = std::chrono::duration<float, std::milli>(end - start).count();

	if (current_backend != BackendType::CodeGen) // no need to compile if we are in codegen mode
	{
		auto start_time = chrono::high_resolution_clock::now();
		CompileAndLoadKernelModule(program, program_id);
		auto end_time = chrono::high_resolution_clock::now();
		host_compile_time = chrono::duration<float, std::milli>(end_time - start_time).count();

		start_time = chrono::high_resolution_clock::now();
		CompileKernels(program);
		end_time = chrono::high_resolution_clock::now();
		shader_compile_time = chrono::duration<float, std::milli>(end_time - start_time).count();
	}
}

vector<TFTensor*> TensorProgram::Evaluate(
    const vector<TFTensor*>& input) const {
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
	properties += "  Host readbacks: " + to_string(ir.readbacks) + "\n";
	properties += "  Host writes: " + to_string(ir.writebacks) + "\n";
	properties += "  Lines of generated code: " + to_string(lines) + "\n";
	properties += "  IR Compile time: " + to_string(compile_time) + " ms\n";
	properties += "  Codegen time: " + to_string(codegen_time) + " ms\n";
	if(host_compile_time > 0.01f)
		properties += "  Host Compile time: " + to_string(host_compile_time) + " ms\n";
	if (shader_compile_time > 0.01f)
		properties += "  Shader Compile time: " + to_string(shader_compile_time) + " ms\n";
	return properties;
}

size_t TensorProgram::program_id = 0;

}  // namespace TensorFrost