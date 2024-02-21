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

	vector<Node*> memory_inputs = program->ir_->memory_inputs;
	unordered_map<Node*, unordered_map<int, Node*>> shape_memory_map = program->ir_->shape_memory_map;
	unordered_map<int, Node*> output_memory_map = program->ir_->output_memory_map;
	int output_count = output_memory_map.size();

	if (memory_inputs.size() != inputs.size()) {
		throw std::runtime_error(
		    "Invalid number of inputs for TensorProgram. Expected " +
		    to_string(memory_inputs.size()) + ", got " + to_string(inputs.size()));
	}

	if (output_count == 0) {
		throw std::runtime_error("TensorProgram does not do any computation: no outputs");
	}

	vector<uint> input_offsets;
	unordered_set<Node*> processed_shapes;
	vector<TensorMemory*> to_remove;
	for (int i = 0; i < memory_inputs.size(); i++) {
		// add and check input shapes
		Node* input = memory_inputs[i];
		unordered_map<int, Node*> args = shape_memory_map[input];
		vector<int> shape = inputs[i]->GetShape();

		if (shape.size() != args.size()) {
			throw std::runtime_error(
			    "Invalid dimension for input " + to_string(i) + ". Expected " +
				to_string(args.size()) + ", got " + to_string(shape.size()));
		}

		//check if data type is correct
		if (input->tensor_->type != inputs[i]->type) {
			throw std::runtime_error(
			    "Invalid data type for input " + to_string(i) + ". Expected " +
			                         DataTypeToString(input->tensor_->type) +
			                         ", got " + DataTypeToString(inputs[i]->type));
		}

		for (int j = 0; j < args.size(); j++) {
			Node* shape_node = args[j];
			
			// if shape node is a constant, compare constant value to input shape
			int expected = -1;

			if (shape_node->name == "const") 
			{
				expected = shape_node->GetTensor()->data[0];
			}
			else if (shape_node->memory_type_ == MemoryType::Shape) 
			{
				if (!processed_shapes.contains(shape_node)) //if not already allocated
				{
					// Allocate scalar for shape
					vector<uint> data;
					data.push_back(shape[j]);
					TensorMemory* shape_mem =
					    global_memory_manager->AllocateWithData({1}, data);
					// add input memory shape offset
					input_offsets.push_back(shape_mem->frame->start);
					to_remove.push_back(shape_mem);
					processed_shapes.insert(shape_node);
				}
			}
			else 
			{
				throw std::runtime_error("Invalid shape node type for input " + to_string(i));
			}

			if (expected != -1 && expected != shape[j])
			{
				throw std::runtime_error("Invalid input shape " + to_string(j) +
				                         " for input " + to_string(i) + ". Expected " +
				                         to_string(expected) + ", got " +
				                         to_string(shape[j]));
			}
		}

		// add input memory offset
		input_offsets.push_back(inputs[i]->frame->start);
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
		outputs[i]->type = output_memory_map[i]->tensor_->type;
	}

	for (auto mem : to_remove)
	{
		delete mem;
	}

	return outputs;
}

}  // namespace TensorFrost