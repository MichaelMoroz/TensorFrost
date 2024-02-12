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

	auto allocator = [&](uint*& mem, uint* a, uint dim) { 
		vector<int> shape;
		for (int i = 0; i < dim; i++)
		{
			shape.push_back(a[i]);
		}
		uint off = global_memory_manager->Allocate(shape)->frame->start;
		mem = ((CpuMemoryManager*)global_memory_manager)->memory.data();
		return off;
	};

	auto deallocator = [&](uint a) { 
		global_memory_manager->Free(a);
	};

	program->execute_callback(in, out, mem, allocator, deallocator);

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

	//unordered_set<TensorMemory*> temp_memory;
	//map<int, TensorMemory*> output_memory;
	//// go over the kernels and execute them
	//for (auto& i : program->kernels_) {
	//	Kernel* kernel = &i;
	//
	//	switch (kernel->type_) {
	//		case KernelType::Memory: {
	//			Node* node = kernel->begin_;
	//			// get shape arguments
	//			ArgMap args = node->GetArgumentMap(Arg::Shape);
	//			uint dims = MaxIndexCount(args);
	//			// get shape from shape constants
	//			vector<int> shape;
	//			for (int i = 0; i < dims; i++) {
	//				Node* shape_node = args[i]->from_->get();
	//				if (shape_node->name == "const") {
	//					shape.push_back(shape_node->GetTensor()->data[0]);
	//					continue;
	//				}
	//				if (!shape_constants.contains(shape_node)) {
	//					throw std::runtime_error("Shape constant not found");
	//				}
	//				shape.push_back(shape_constants[shape_node]);
	//			}
	//			TensorMemory* memory = global_memory_manager->Allocate(shape);
	//			if (node->memory_type_ != MemoryType::Output) {
	//				temp_memory.insert(memory);
	//			} else {
	//				output_memory[node->memory_index_] = memory;
	//			}
	//			memory_map[node] = memory;
	//		} break;
	//		case KernelType::Compute: {
	//			Node* begin = kernel->begin_;
	//			ArgMap shape = kernel->shape;
	//			int dim = kernel->dim;
	//			vector<uint> shape_values(dim);
	//			int thread_count = 1;
	//
	//			for (int i = 0; i < dim; i++) 
	//			{
	//				const Arg* arg = shape[i];
	//				int val = 0;
	//				if (arg->from_->get()->name == "const") 
	//				{
	//					val = arg->from_->get()->GetTensor()->data[0];
	//				}
	//				else
	//				{
	//					val = shape_constants[arg->from_->get()];
	//				}
	//
	//				thread_count *= val;
	//				shape_values[i] = val;
	//			}
	//
	//			if (kernel->indexing_mode_ == KernelIndexingMode::Linear)
	//			{
	//				shape_values[0] = thread_count;
	//			}
	//
	//			vector<uint> memory_offsets;
	//			memory_offsets.resize(kernel->memory.size());
	//			vector<uint> variables;
	//			variables.resize(kernel->variables.size());
	//			for (auto& j : kernel->memory) {
	//				memory_offsets[j.second] = memory_map[j.first]->frame->start;
	//			}
	//			for (auto& j : kernel->variables) {
	//				// if variable is a constant, add constant value to variable offsets
	//				if (j.first->name == "const") {
	//					variables[j.second] = j.first->GetTensor()->data[0];
	//				} else {
	//					if (shape_constants.contains(j.first)) {
	//						variables[j.second] = shape_constants[j.first];
	//					} else {
	//						// otherwise, load variable from memory
	//						uint offset = memory_map[j.first]->frame->start;
	//						uint variable =
	//						    ((CpuMemoryManager*)global_memory_manager)->memory[offset];
	//						variables[j.second] = variable;
	//					}
	//				}
	//			}
	//
	//			kernel->execute_callback(global_memory_manager, variables,
	//			                         memory_offsets, shape_values);
	//		} break;
	//	}
	//}
	//
	//// delete allocated tensormemory
	//for (auto it = temp_memory.begin(); it != temp_memory.end(); ++it) {
	//	delete *it;
	//}
	//
	//vector<TensorMemory*> outputs;
	//outputs.reserve(output_memory.size());
	//for (auto i : output_memory) {
	//	outputs.push_back(i.second);
	//}
	//return outputs;
}

}  // namespace TensorFrost