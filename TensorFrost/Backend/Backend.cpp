#include "Backend.h"

namespace TensorFrost {

BackendType current_backend = BackendType::NotInitialized;
CodeGenLang current_kernel_lang = CodeGenLang::CPP;
CodeGenLang current_main_lang = CodeGenLang::CPP;

void InitializeBackend(BackendType backendType, const string& compilerOptions, CodeGenLang kernelType) {
	if (current_backend != BackendType::NotInitialized) {
		cout << "Warning: Backend already initialized, stopping current backend\n" << endl;

		switch (current_backend) {
			case BackendType::CPU:
				break;
			case BackendType::Vulkan:
				break;
			case BackendType::OpenGL:
				StopOpenGL();
				break;
		}
	}

	if (!compilerOptions.empty()) {
		kernel_compile_options = compilerOptions;
	}

	current_backend = backendType;

	switch (backendType) {
		case BackendType::CPU:
		case BackendType::CodeGen:
			current_kernel_lang = CodeGenLang::CPP;
			global_memory_manager = new CpuMemoryManager();
			global_kernel_manager = new CpuKernelManager();
			break;
		case BackendType::Vulkan:
			throw std::runtime_error("Vulkan backend not implemented yet");
			current_kernel_lang = CodeGenLang::GLSL;
			break;
		case BackendType::OpenGL:
			StartOpenGL();
			current_kernel_lang = CodeGenLang::GLSL;
			global_memory_manager = new OpenGLMemoryManager();
			global_kernel_manager = new OpenGLKernelManager();
			break;
	}

	if (kernelType != CodeGenLang::None) {
		current_kernel_lang = kernelType;
	}
}

void CompileKernels(Program* program) {
	for(auto& kernel : program->kernels_) {
		switch (current_backend) {
			case BackendType::CPU:
				//already in the host program
				break;
			case BackendType::Vulkan:
				throw std::runtime_error("Vulkan backend not implemented yet");
			case BackendType::OpenGL:
				((OpenGLKernelManager*)global_kernel_manager)->CompileKernel(&kernel);
				break;
			default:
				throw std::runtime_error("Backend not implemented");
		}
	}
}

TensorProp GetTensorProp(TensorMemory* tensor) {
	TensorProp prop;
	prop.offset = tensor->frame->start;
	auto shape = tensor->GetShape();
	prop.dim = (uint)shape.size();
	prop.shape = new uint[prop.dim];
	for (uint i = 0; i < prop.dim; i++) {
		prop.shape[i] = (uint)shape[i];
	}
	prop.type = tensor->type;
	return prop;
}

TensorProp Allocator(uint* a, uint dim, DataType type) {
	vector<int> shape;
	for (uint i = 0; i < dim; i++) {
		shape.push_back(a[i]);
	}
	TensorMemory* tensor = global_memory_manager->Allocate(shape, type);
	return GetTensorProp(tensor);
}

void Deallocator(TensorProp a) { 
	global_memory_manager->Free(a.offset); 
	delete[] a.shape;
}

uint Readback(TensorProp a, uint b) {
	return global_memory_manager->ReadbackValue(global_memory_manager->allocated_by_offset[a.offset], b);
}

void Writeback(TensorProp a, uint b, uint c) {
	global_memory_manager->WritebackValue(global_memory_manager->allocated_by_offset[a.offset], b, c);
}

void Dispatch(DispatchInfo info) {
	global_kernel_manager->DispatchKernel(info);
}

vector<TensorMemory*> ExecuteProgram(
    Program* program, vector<TensorMemory*> inputs) {

	if (current_backend == BackendType::CodeGen) {
		throw std::runtime_error("Cannot execute program with code generation backend");
	}

	int memory_input_count = (int)program->ir_->memory_inputs.size();

	if (memory_input_count != inputs.size()) {
		throw std::runtime_error(
		    "Invalid number of inputs for TensorProgram. Expected " +
		    to_string(memory_input_count) + ", got " + to_string(inputs.size()));
	}

	vector<TensorProp> input_tensors;
	for (int i = 0; i < memory_input_count; i++) {
		// add input memory offset
		input_tensors.push_back(GetTensorProp(inputs[i]));
	}

	unordered_map<int, Node*> output_memory_map = program->ir_->output_memory_map;
	int output_count = (int)output_memory_map.size();

	TensorProp* in = input_tensors.data();
	TensorProp* out = new TensorProp[output_count];

	if (current_backend == BackendType::OpenGL) {
		StartDebugRegion(program->program_name);
	}

	program->execute_callback(in, out, Allocator, Deallocator, Readback, Writeback, Dispatch);

	if (current_backend == BackendType::OpenGL) {
		EndDebugRegion();
		//Finish();
	}

	vector<TensorMemory*> outputs;
	outputs.resize(output_count);
	for (int i = 0; i < output_count; i++)
	{
		outputs[i] = global_memory_manager->allocated_by_offset[out[i].offset];
		//TODO this can potentially overwite some of the tensor properties used elsewhere
		outputs[i]->shape.clear();
		for (uint j = 0; j < out[i].dim; j++) {
			outputs[i]->shape.push_back(out[i].shape[j]);
		}
		outputs[i]->type = out[i].type;
	}

	return outputs;
}

}  // namespace TensorFrost