#include "Backend.h"

namespace TensorFrost {

TensorMemoryManager* global_memory_manager = nullptr;

void InitializeBackend(BackendType backendType, const string& compilerOptions) {
	if (!compilerOptions.empty()) {
		kernel_compile_options = compilerOptions;
	}
	switch (backendType) {
		case BackendType::CPU:
			global_memory_manager = new CpuMemoryManager();
			break;
		case BackendType::WGPU:
			throw std::runtime_error("WGPU backend not implemented yet");
			break;
	}
}

uint Allocator(uint*& mem, uint* a, uint dim) {
	vector<int> shape;
	for (int i = 0; i < dim; i++) {
		shape.push_back(a[i]);
	}
	uint off = global_memory_manager->Allocate(shape)->frame->start;
	mem = ((CpuMemoryManager*)global_memory_manager)->memory.data();
	return off;
}

void Deallocator(uint a) { global_memory_manager->Free(a); }

vector<TensorMemory*> ExecuteProgram(
    Program* program, vector<TensorMemory*> inputs) {
	vector<Node*> memory_inputs;

	for (auto node = program->ir_->begin(); !node.is_end(); ++node) {
		if (node->memory_type_ == MemoryType::Input) {
			memory_inputs.push_back(*node);
		}
	}

	if (memory_inputs.size() != inputs.size()) {
		throw std::runtime_error(
		    "Invalid number of inputs for TensorProgram. Expected " +
		    to_string(memory_inputs.size()) + ", got " + to_string(inputs.size()));
	}

	map<Node*, TensorMemory*> memory_map;
	map<Node*, int> shape_constants;
	for (int i = 0; i < memory_inputs.size(); i++) {
		memory_map[memory_inputs[i]] = inputs[i];
		// get shape arguments
		Arguments args = memory_inputs[i]->GetArguments(Arg::Shape);
		vector<int> shape = inputs[i]->GetShape();
		for (int j = 0; j < args.size(); j++) {
			Node* shape_node = args[j].from_->get();
			// if shape node is a constant, compare constant value to input shape
			bool invalid_shape = false;
			int expected = -1;
			bool is_const = false;

			if (shape_node->name == "const") {
				expected = shape_node->GetTensor()->data[0];
				is_const = true;
			}

			if (shape_constants.contains(shape_node)) {
				expected = shape_constants[shape_node];
			}

			if (expected != -1 && expected != shape[j])
			{
				throw std::runtime_error("Invalid input shape " + to_string(j) +
				                         " for input " + to_string(i) + ". Expected " +
				                         to_string(expected) + ", got " +
				                         to_string(shape[j]));
			}

			shape_constants[shape_node] = shape[j];
		}
	}

	vector<uint> input_offsets;
	vector<TensorMemory*> to_remove;
	int output_count = 0;

	for (auto node = program->ir_->begin(); !node.is_end(); ++node) {
		if (node->memory_type_ == MemoryType::Input) {
			input_offsets.push_back(memory_map[*node]->frame->start);
		}
		if (node->memory_type_ == MemoryType::Shape)
		{
			//Allocate scalar 
			vector<uint> data;
			data.push_back(shape_constants[*node]);
			TensorMemory* shape_mem =
			    global_memory_manager->AllocateWithData({1}, data);
			input_offsets.push_back(shape_mem->frame->start);
			to_remove.push_back(shape_mem);
		}
		if (node->memory_type_ == MemoryType::Output)
		{
			output_count++;
		}
	}

	uint* mem = ((CpuMemoryManager*)global_memory_manager)->memory.data();
	uint* in = input_offsets.data();
	uint* out = new uint[output_count];

	program->execute_callback(in, out, mem, Allocator, Deallocator);

	vector<TensorMemory*> outputs;
	outputs.resize(output_count);
	for (int i = 0; i < output_count; i++)
	{
		outputs[i] = global_memory_manager->allocated_by_offset[out[i]];
	}

	for (auto mem : to_remove)
	{
		delete mem;
	}

	return outputs;
}

}  // namespace TensorFrost