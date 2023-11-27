#include "Backend.h"

namespace TensorFrost {

TensorMemoryManager* global_memory_manager = nullptr;

void InitializeBackend(BackendType backendType, const string& compilerPath) {
	if (!compilerPath.empty()) {
		c_compiler_path = compilerPath;
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

vector<TensorMemory*> TensorFrost::ExecuteProgram(
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
		Arguments args = memory_inputs[i]->GetArguments(Argument::Shape);
		vector<int> shape = inputs[i]->GetShape();
		for (int j = 0; j < args.size(); j++) {
			Node* shape_node = args[j].from_->get();
			// if shape node is a constant, compare constant value to input shape
			if (shape_node->name == "const") {
				int val = shape_node->tensor_->data[0];
				if (val != shape[j]) {
					throw std::runtime_error("Invalid input shape " + to_string(j) +
					                         " for input " + to_string(i) +
					                         ". Expected " + to_string(val) + ", got " +
					                         to_string(shape[j]));
				}
			}

			shape_constants[shape_node] = shape[j];
		}
	}

	unordered_set<TensorMemory*> temp_memory;
	map<int, TensorMemory*> output_memory;
	// go over the kernels and execute them
	for (auto& i : program->kernels_) {
		Kernel* kernel = &i;

		switch (kernel->type_) {
			case KernelType::Memory: {
				Node* node = kernel->begin_;
				// get shape arguments
				Arguments args = node->GetArguments(Argument::Shape);
				// get shape from shape constants
				vector<int> shape;
				for (auto& arg : args) {
					Node* shape_node = arg.from_->get();
					shape.push_back(shape_constants[shape_node]);
				}
				TensorMemory* memory = global_memory_manager->Allocate(shape);
				if (node->memory_type_ != MemoryType::Output) {
					temp_memory.insert(memory);
				} else {
					output_memory[node->memory_index_] = memory;
				}
				memory_map[node] = memory;
			} break;
			case KernelType::Compute: {
				Node* begin = kernel->begin_;
				map<int, Tensor*> args = begin->GetArgumentTensors(Argument::Shape);
				int thread_count = 1;
				for (auto& arg : args) {
					thread_count *= shape_constants[arg.second->node];
				}

				vector<uint> memory_offsets;
				memory_offsets.resize(kernel->memory.size());
				vector<uint> variables;
				variables.resize(kernel->variables.size());
				for (auto& j : kernel->memory) {
					memory_offsets[j.second] = memory_map[j.first]->frame->start;
				}
				for (auto& j : kernel->variables) {
					// if variable is a constant, add constant value to variable offsets
					if (j.first->name == "const") {
						variables[j.second] = j.first->tensor_->data[0];
					} else {
						if (shape_constants.contains(j.first)) {
							variables[j.second] = shape_constants[j.first];
						} else {
							// otherwise, load variable from memory
							uint offset = memory_map[j.first]->frame->start;
							uint variable =
							    ((CpuMemoryManager*)global_memory_manager)->memory[offset];
							variables[j.second] = variable;
						}
					}
				}

				kernel->execute_callback(global_memory_manager, variables,
				                         memory_offsets, thread_count);
			} break;
		}
	}

	// delete allocated tensormemory
	for (auto it = temp_memory.begin(); it != temp_memory.end(); ++it) {
		delete *it;
	}

	vector<TensorMemory*> outputs;
	outputs.reserve(output_memory.size());
	for (auto i : output_memory) {
		outputs.push_back(i.second);
	}
	return outputs;
}

}  // namespace TensorFrost