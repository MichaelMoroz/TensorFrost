#include "Backend.h"

namespace TensorFrost {

TensorMemoryManager* global_memory_manager = nullptr;

void InitializeBackend(BackendType backendType, const string& compilerOptions) {
	if (!compilerOptions.empty()) {
		kernel_compile_options = compilerOptions;
	}

	current_backend = backendType;

	switch (backendType) {
		case BackendType::CPU:
			global_memory_manager = new CpuMemoryManager();
			break;
		case BackendType::Vulkan:
			throw std::runtime_error("Vulkan backend not implemented yet");
			break;
		case BackendType::OpenGL:
			throw std::runtime_error("OpenGL backend not implemented yet");
			break;
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

TensorProp Allocator(uint*& mem, uint* a, uint dim, DataType type) {
	vector<int> shape;
	for (uint i = 0; i < dim; i++) {
		shape.push_back(a[i]);
	}
	TensorMemory* tensor = global_memory_manager->Allocate(shape, type);
	TensorProp prop = GetTensorProp(tensor);
	mem = ((CpuMemoryManager*)global_memory_manager)->memory.data();
	return prop;
}

void Deallocator(TensorProp a) { 
	global_memory_manager->Free(a.offset); 
	delete[] a.shape;
}

uint Readback(TensorProp a, uint b) {
	global_memory_manager->ReadbackValue(global_memory_manager->allocated_by_offset[a.offset], b);
}

void Writeback(TensorProp a, uint b, uint c) {
	global_memory_manager->WritebackValue(global_memory_manager->allocated_by_offset[a.offset], b, c);
}

void Dispatch(int kernel_id, TensorProp* inputs, uint* variables, uint* shape) {

}

vector<TensorMemory*> ExecuteProgram(
    Program* program, vector<TensorMemory*> inputs) {

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

	uint* mem = ((CpuMemoryManager*)global_memory_manager)->memory.data();
	TensorProp* in = input_tensors.data();
	TensorProp* out = new TensorProp[output_count];

	program->execute_callback(in, out, Allocator, Deallocator, Readback, Writeback, Dispatch);

	vector<TensorMemory*> outputs;
	outputs.resize(output_count);
	for (int i = 0; i < output_count; i++)
	{
		outputs[i] = global_memory_manager->allocated_by_offset[out[i].offset];
		outputs[i]->type = output_memory_map[i]->tensor_->type;
	}

	return outputs;
}

}  // namespace TensorFrost