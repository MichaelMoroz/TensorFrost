#include "Compiler/KernelGen.h"

namespace TensorFrost {

Program* GenerateProgram(IR* ir) 
{
	ir->CompileIR(); 

	auto* program = new Program(ir);

	vector<Node*> kernels = ir->GetNodesOfType("kernel");

	for (auto kernel : kernels)
	{
		// get the kernel type
		map<Node*, size_t> variables;
		map<Node*, bool> read_write;
		set<Node*> group_memory;
		NodeArguments shape = kernel->args.GetArguments(ArgType::Shape);
		size_t variable_index = 0;

		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			if(node->name == "group_memory") {
				group_memory.insert(*node);
				continue;
			}
			if (node->op->HasAllTypes(OpProp::MemoryOp)) {
				//if the memory is inside of this kernel - skip node
				if (node->flags.has(NodeProp::LocalMemoryOp)) {
					continue;
				}

				// get the memory node
				const Tensor* memory = node->args.GetTensor(ArgType::Memory);

				if(node->op->HasAllTypes(OpProp::Modifier)) {
					read_write[memory->node_] |= true;
				} else {
					read_write[memory->node_] |= false;
				}
			}

			// get all input arguments
			for (auto [id, from] : node->args.Inputs()) {
				if (id.first == ArgType::Input)
				{
					bool from_outside_kernel = !from->HasParent(kernel);
					if (from_outside_kernel && !variables.contains(from)) {
						variables[from] = variable_index++;
					}
				}
			}
		}

		map<Node*, size_t> read_write_memory;
		map<Node*, size_t> read_only_memory;
		size_t read_write_index = 0;
		size_t read_only_index = 0;
		for(auto [node, rw] : read_write) {
			if(rw) {
				read_write_memory[node] = read_write_index++;
			} else {
				read_only_memory[node] = read_only_index++;
			}
		}

		// add the kernel to the program
		program->AddKernel(kernel, variables, read_write_memory, read_only_memory, group_memory, shape);
	}

	return program;
}

}  // namespace TensorFrost