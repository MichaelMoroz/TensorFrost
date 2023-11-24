#include "Backend.h"

namespace TensorFrost {

TensorMemoryManager* GlobalMemoryManager = nullptr;

void InitializeBackend(BackendType backendType, string compilerPath) 
{
	if (compilerPath != "") {
		C_COMPILER_PATH = compilerPath;
	}
	switch (backendType) {
		case BackendType::CPU:
			GlobalMemoryManager = new CPU_MemoryManager();
			break;
		case BackendType::WGPU:
			throw std::runtime_error("WGPU backend not implemented yet");
			break;
	}
}

vector<TensorMemory*> TensorFrost::ExecuteProgram(Program* program, vector<TensorMemory*> inputs) 
{
	vector<Node*> memory_inputs;

	for (auto node = program->ir_->begin(); !node.is_end(); ++node) {
		if (node->memory_type_ == MemoryType::Input) {
			memory_inputs.push_back(*node);
		}
	}

	if(memory_inputs.size() != inputs.size()) {
		throw std::runtime_error("Invalid number of inputs for TensorProgram. Expected " + to_string(memory_inputs.size()) + ", got " + to_string(inputs.size()));
	}

	map<Node*, TensorMemory*> memory_map;
	map<Node*, int> shape_constants;
	for (int i = 0; i < memory_inputs.size(); i++) {
		memory_map[memory_inputs[i]] = inputs[i];
		//get shape arguments
		Arguments args = memory_inputs[i]->GetArguments(Argument::Shape);
		vector<int> shape = inputs[i]->GetShape();
		for (int j = 0; j < args.size(); j++) {
			Node* shape_node = args[j].from_->get();
			//if shape node is a constant, compare constant value to input shape
			if (shape_node->name == "const") 
			{
				int val = shape_node->tensor_->data[0];
				if (val != shape[j]) {
					throw std::runtime_error("Invalid input shape " + to_string(j) + " for input " + to_string(i) + ". Expected " + to_string(val) + ", got " + to_string(shape[j]));
				}
			}
			
			shape_constants[shape_node] = shape[j];
		}
	}

	unordered_set<TensorMemory*> temp_memory;
	map<int, TensorMemory*> output_memory;
	//go over the kernels and execute them
	for (int i = 0; i < program->kernels_.size(); i++) {
		Kernel* kernel = &program->kernels_[i];
		
		switch (kernel->type_) {
			case KernelType::Memory:
				{
					Node* node = kernel->begin_;
					// get shape arguments
					Arguments args = node->GetArguments(Argument::Shape);
					// get shape from shape constants
					vector<int> shape;
					for (int j = 0; j < args.size(); j++) {
						Node* shape_node = args[j].from_->get();
						shape.push_back(shape_constants[shape_node]);
					}
					TensorMemory* memory = GlobalMemoryManager->Allocate(shape);
					if (node->memory_type_ != MemoryType::Output) {
						temp_memory.insert(memory);
					}
					else {
						output_memory[node->memory_index_] = memory;
					}
					memory_map[node] = memory;
				}
				break;
			case KernelType::Compute:
				{
					Node* begin = kernel->begin_;
					map<int, Tensor*> args = begin->GetArgumentTensors(Argument::Shape);
					int thread_count = 1;
					for (int j = 0; j < args.size(); j++) {
						thread_count *= shape_constants[args[j]->node];
					}

					vector<uint> memory_offsets;
					vector<uint> variables;
					for (int j = 0; j < kernel->memory.size(); j++) {
					  memory_offsets.push_back(
					      memory_map[kernel->memory[j]]->frame->start);
					}
					for (int j = 0; j < kernel->variables.size(); j++) {
						//if variable is a constant, add constant value to variable offsets
						if (kernel->variables[j]->name == "const") {
							variables.push_back(kernel->variables[j]->tensor_->data[0]);
						}
						else 
						{
							if (shape_constants.contains(kernel->variables[j])) {
								variables.push_back(shape_constants[kernel->variables[j]]);
							}
							else
							{
								// otherwise, load variable from memory
								uint offset = memory_map[kernel->variables[j]]->frame->start;
								uint variable =
									((CPU_MemoryManager*)GlobalMemoryManager)->memory[offset];
								variables.push_back(variable);
							}
						}
					}

					kernel->execute_callback(GlobalMemoryManager, variables, memory_offsets, thread_count);
				}
				break;
		}
	}

	//delete allocated tensormemory
	for (auto it = temp_memory.begin(); it != temp_memory.end(); ++it) {
		delete *it;
	}

	vector<TensorMemory*> outputs;
	for (int i = 0; i < output_memory.size(); i++) {
		outputs.push_back(output_memory[i]);
	}
	return outputs;
}

}// namespace TensorFrost