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

TFTensor Allocator(const char* name, const size_t* a, size_t dim, TFType type, void* data) {
	vector<size_t> shape(a, a + dim);
	return *global_memory_manager->AllocateTensor(shape, type, name);
}

void Deallocator(TFTensor a, void* data) {
	global_memory_manager->DeallocateTensor(a);
}

uint Readback(TFTensor a, size_t index, void* data) {
	return global_memory_manager->ReadbackValue(&a, index);
}

void Writeback(TFTensor a, size_t index, uint32_t value, void* data) {
	global_memory_manager->WritebackValue(&a, index, value);
}

void Dispatch(TFDispatchInfo info, void* data) {
	global_kernel_manager->DispatchKernel(info);
}

void Region(const char* name, bool begin, void* data) {
	if (current_backend == BackendType::OpenGL) {
		if (begin) {
			StartDebugRegion(name);
		} else {
			EndDebugRegion();
		}
	}
}

vector<TFTensor*> ExecuteProgram(
    Program* program, vector<TFTensor*> inputs) {

	if (current_backend == BackendType::CodeGen) {
		throw std::runtime_error("Cannot execute program with code generation backend");
	}

	int memory_input_count = (int)program->ir_->input_memory_map.size();

	if (memory_input_count != inputs.size()) {
		throw std::runtime_error(
		    "Invalid number of inputs for TensorProgram. Expected " +
		    to_string(memory_input_count) + ", got " + to_string(inputs.size()));
	}

	vector<TFTensor> input_tensors;
	for (int i = 0; i < memory_input_count; i++) {
		input_tensors.push_back(*inputs[i]);
	}

	unordered_map<int, Node*> output_memory_map = program->ir_->output_memory_map;
	int output_count = (int)output_memory_map.size();

	TFTensor* in = input_tensors.data();
	TFTensor* out = new TFTensor[output_count];

	program->execute_callback(in, out, {Allocator, Deallocator, Readback, Writeback, Dispatch, Region, nullptr});

	vector<TFTensor*> outputs = vector<TFTensor*>(output_count);
	for (int i = 0; i < output_count; i++) {
		outputs[i] = &out[i];
	}

	return outputs;
}

}  // namespace TensorFrost