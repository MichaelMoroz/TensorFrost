#include "Backend.h"

namespace TensorFrost {

BackendType current_backend = BackendType::NotInitialized;
CodeGenLang current_kernel_lang = CodeGenLang::CPP;
CodeGenLang current_main_lang = CodeGenLang::CPP;
bool strip_debug_names = false;

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
			default:
				throw std::runtime_error("Backend not implemented");
		}
	}

	if (!compilerOptions.empty()) {
		kernelCompileOptions = compilerOptions;
	} else if(backendType != BackendType::CPU) {
		kernelCompileOptions = ""; //no need for cpu optimizations on other backends
	} else {
#ifdef _WIN32
		kernelCompileOptions = "/O2 /fp:fast /openmp";
#else
		kernelCompileOptions = "-O3 -ffast-math -fopenmp";
#endif
	}

#ifdef _DEBUG
#ifdef _WIN32
	kernelCompileOptions = "/Zi";
#else
	kernelCompileOptions = "-g";
#endif
#endif

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
		default:
			throw std::runtime_error("Backend not implemented");
	}

	if (kernelType != CodeGenLang::None) {
		current_kernel_lang = kernelType;
	}
}

void CompileKernels(Program* program) {
	auto start_time = chrono::high_resolution_clock::now();
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
	auto end_time = chrono::high_resolution_clock::now();
	float milliseconds = chrono::duration<float, std::milli>(end_time - start_time).count();
	program->shader_compile_time = milliseconds;
}

TFTensor Allocator(const char* name, const size_t* a, size_t dim, TFDataFormat format, void* data) {
	try {
		vector<size_t> shape(a, a + dim);
		return *global_memory_manager->AllocateTensor(shape, format, name);
	} catch (const std::exception& e) {
		size_t size = 1;
		for (size_t i = 0; i < dim; i++) {
			size *= a[i];
		}
		throw std::runtime_error("Error allocating tensor " + string(name) + ": " + e.what() + ", requested size: " + to_string(size));
	}
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

//#define PROFILE_EXECUTION

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

#ifdef PROFILE_EXECUTION
	auto start = chrono::high_resolution_clock::now();
#endif
	try {
		program->execute_callback(in, out, {Allocator, Deallocator, Readback, Writeback, Dispatch, Region, nullptr});
	} catch (const std::exception& e) {
		throw std::runtime_error("Error executing program " + program->program_name + ": " + e.what());
	}

#ifdef PROFILE_EXECUTION
	Finish();
	auto end = chrono::high_resolution_clock::now();
	float milliseconds = chrono::duration<float, std::milli>(end - start).count();
	program->last_execution_time = milliseconds;
#endif

	vector<TFTensor*> outputs = vector<TFTensor*>(output_count);
	for (int i = 0; i < output_count; i++) {
		outputs[i] = &out[i];
	}

	return outputs;
}

}  // namespace TensorFrost