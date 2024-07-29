#include "Compiler/KernelGen.h"

namespace TensorFrost {

bool isConstantAndEqualTo(const Tensor* tensor, float value) {
	if (tensor->node_->name != "const" || tensor->node_->flags.has(NodeProp::Modified)) {
		return false;
	}

	switch (tensor->node_->type) {
		case TFType::Float:
			return AsFloat(tensor->node_->data[0]) == value;
		case TFType::Int:
			return AsInt(tensor->node_->data[0]) == value;
		case TFType::Uint:
			return tensor->node_->data[0] == value;
		default:
			throw std::runtime_error("Unexpected type in isConstantAndEqualTo");
	}
}

bool isConstant(const Tensor* tensor) {
	return tensor->node_->name == "const" && !tensor->node_->flags.has(NodeProp::Modified);
}

Tensor* ApplyMultiOP(const Tensor* a, const Tensor* b, std::function<float(float, float)> opF32, std::function<int(int, int)> opI32, std::function<uint(uint, uint)> opU32) {
	switch (a->node_->type) {
		case TFType::Float:
			return &Tensor::Constant(opF32(AsFloat(a->node_->data[0]), AsFloat(b->node_->data[0])));
		case TFType::Int:
			return &Tensor::Constant(opI32(AsInt(a->node_->data[0]), AsInt(b->node_->data[0])));
		case TFType::Uint:
			return &Tensor::Constant(opU32(a->node_->data[0], b->node_->data[0]));
		default:
			throw std::runtime_error("Unexpected type in ApplyMultiOP");
	}
}

Tensor* ApplyUnaryOP(const Tensor* a, std::function<float(float)> opF32, std::function<int(int)> opI32, std::function<uint(uint)> opU32) {
	switch (a->node_->type) {
		case TFType::Float:
			return &Tensor::Constant(opF32(AsFloat(a->node_->data[0])));
		case TFType::Int:
			return &Tensor::Constant(opI32(AsInt(a->node_->data[0])));
		case TFType::Uint:
			return &Tensor::Constant(opU32(a->node_->data[0]));
		default:
			throw std::runtime_error("Unexpected type in ApplyUnaryOP");
	}
}

#define ApplyOP(v1, v2, op) ApplyMultiOP(v1, v2, [](float a, float b) { return a op b; }, [](int a, int b) { return a op b; }, [](uint a, uint b) { return a op b; })
#define ApplyFUNC(v1, v2, func) ApplyMultiOP(v1, v2, [](float a, float b) { return func(a, b); }, [](int a, int b) { return func(a, b); }, [](uint a, uint b) { return func(a, b); })
#define ApplyUOP(v1, op) ApplyUnaryOP(v1, [](float a) { return op a; }, [](int a) { return op a; }, [](uint a) { return op a; })
#define ApplyUFUNC(v1, func) ApplyUnaryOP(v1, [](float a) { return func(a); }, [](int a) { return func(a); }, [](uint a) { return func(a); }

void IR::OptimizeOperations()
{
	for (auto node = begin(); !node.end(); node.next()) {
		//get node operation
		const string op = node->name;

		//get inputs
		map<int, const Tensor*> inputs = node->args.GetTensors(ArgType::Input);
		ExecuteExpressionAfter(*node, [&]() {
			const Tensor* result = nullptr;
			if (op == "add") {
				// if any are zero, replace with the other
				if (isConstantAndEqualTo(inputs[0], 0.0F)) {
					// replace with input 1
					result = inputs[1];
				} else if (isConstantAndEqualTo(inputs[1], 0.0F)) {
					// replace with input 0
					result = inputs[0];
				}

				// if all are constants, replace with result
				if (isConstant(inputs[0]) && isConstant(inputs[1])) {
					// replace with result
					result = ApplyOP(inputs[0], inputs[1], +);
				}
			} else if (op == "sub") {
				// if any are zero, replace with the other
				if (isConstantAndEqualTo(inputs[0], 0.0F)) {
					// replace with negation of input 1
					result = &(-*inputs[1]);
				} else if (isConstantAndEqualTo(inputs[1], 0.0F)) {
					// replace with input 0
					result = inputs[0];
				}

				// if all are constants, replace with result
				if (isConstant(inputs[0]) && isConstant(inputs[1])) {
					// compute result
					result = ApplyOP(inputs[0], inputs[1], -);
				}
			} else if (op == "mul") {
				// if any are zero, replace with zero
				if (isConstantAndEqualTo(inputs[0], 0.0F) ||
									    isConstantAndEqualTo(inputs[1], 0.0F)) {
					// replace with zero
					result = &Tensor::Constant(0u, inputs[0]->node_->type);
				}

				// if any are one, replace with the other
				if (isConstantAndEqualTo(inputs[0], 1.0F)) {
					// replace with input 1
					result = inputs[1];
				} else if (isConstantAndEqualTo(inputs[1], 1.0F)) {
					// replace with input 0
					result = inputs[0];
				}

				// if all are constants, replace with result
				if (isConstant(inputs[0]) && isConstant(inputs[1])) {
					// compute result
					result = ApplyOP(inputs[0], inputs[1], *);
				}
			} else if (op == "div") {
				// if first is zero, replace with zero
				if (isConstantAndEqualTo(inputs[0], 0.0F)) {
					// replace with zero
					result = &Tensor::Constant(0u, inputs[0]->node_->type);
				}

				// if second is one, replace with first
				if (isConstantAndEqualTo(inputs[1], 1.0F)) {
					// replace with input 0
					result = inputs[0];
				}

				// if all are constants, replace with result
				if (isConstant(inputs[0]) && isConstant(inputs[1])) {
					// compute result
					result = ApplyOP(inputs[0], inputs[1], /);
				}
			}
			else if (op == "clamp") {
				// if all are constants, replace with result
				if (isConstant(inputs[0]) && isConstant(inputs[1]) && isConstant(inputs[2])) {
					// compute result
					result = ApplyFUNC(inputs[0], inputs[1], max);
					result = ApplyFUNC(result, inputs[2], min);
				}
			}
			else if(op == "neg") {
				if(isConstant(inputs[0])) {
					result = ApplyUnaryOP(inputs[0], [](float a) { return -a; }, [](int a) { return -a; }, [](uint a) { return a; });
				}
			}
			else if(op == "dim_id") { //if the shape of the dimension is 1 then replace with 0
				int dim = node->data[0];
				const Tensor* shape = node->args.Get(ArgType::Shape, dim)->GetTensor();
				if(isConstantAndEqualTo(shape, 1.0F)) {
					result = &Tensor::Constant(0u, TFType::Int);
				}
			}
			//TODO (Moroz): add more optimizations

			// if computed optimized result, replace all node references with it
			if (result != nullptr)
			{
				node->MakeOutputsUseGivenNode(result->node_);
			}
		});
	}
}

void IR::RemoveUnusedOperations() {
	unordered_set<Node*> used_nodes;
	//mark all output nodes as used
	for (auto node = begin(); !node.end(); node.next()) {
		if (node->flags.has(NodeProp::OutputMemory) ||
			node->flags.has(NodeProp::InputMemory) ||
			node->op->HasAllTypes(OpProp::Static)) {
			used_nodes.insert(node.get());
			}
	}

	used_nodes = GetDependencies(used_nodes);

	// remove all nodes that are not used
	unordered_set<Node*> nodes_to_remove;
	for (auto node = begin(); !node.end(); node.next()) {
		if (!used_nodes.contains(node.get())) {
			if (!node->flags.has(NodeProp::InputMemory) && !node->flags.has(NodeProp::OutputMemory))
			{
				nodes_to_remove.insert(node.get());
			}
		}
	}

	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();
}


void IR::RemoveUnusedKernels()
{
	vector<Node*> kernels = GetNodesOfType("kernel");
	vector<Node*> nodes_to_remove;

	for (auto kernel : kernels) {
		// remove all kernel nodes that dont do anything
		int memory_modifiers = 0;
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			if (node->op->HasAllTypes(OpProp::Modifier, OpProp::MemoryOp)) {
				memory_modifiers++;
			}
			//if any output is outside the kernel, then the kernel is needed
			for (auto [edge, to] : node->args.outputs_) {
				auto& [id, from] = edge;
				if (!to->HasParent(kernel)) {
					memory_modifiers++;
				}
			}
		}
		if (memory_modifiers == 0) nodes_to_remove.push_back(kernel);
	}

	// remove all nodes that are not used
	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();
}

#define MAX_UNROLL_SIZE 8
#define MAX_UNROLL_NODES 128
void IR::UnrollLoops()
{
	vector<Node*> loops = GetNodesOfType("loop");

	unordered_set<Node*> loops_to_remove;

	for (auto loop : loops) {
		//get inputs (begin, end, step)
		map<int, const Tensor*> inputs = loop->args.GetTensors(ArgType::Input);

		//try get the constant values
		bool is_const = isConstant(inputs[0]) && isConstant(inputs[1]) && isConstant(inputs[2]);

		if (!is_const) {
			continue;
		}

		int begin = inputs[0]->TryGetConstant();
		int end = inputs[1]->TryGetConstant();
		int step = inputs[2]->TryGetConstant();

		//how many iterations to unroll
		int iters = (end - begin) / step;
		if (iters > MAX_UNROLL_SIZE) {
			continue;
		}

		//get all children of the loop
		vector<Node*> children = GetChildren(loop);

		if (children.size() > MAX_UNROLL_NODES) {
			continue;
		}

		//check if they are not keywords or have no children
		set<Node*> nodes_to_copy;
		bool can_unroll = true;
		for (auto child : children) {
			if (child->op->class_ == OpClass::Keyword || child->child->valid()) {
				can_unroll = false;
				break;
			}
			nodes_to_copy.insert(child);
		}

		if (!can_unroll) {
			continue;
		}

		//unroll the loop
		ExecuteExpressionAfter(loop, [&]() {
			for (int i = begin; i < end; i += step) {
				unordered_map<Node*, Node*> arg_remap;
				Tensor* index = &Tensor::Constant(i);
				//index->SetDebugName(loop->debug_name + "_unroll_" + to_string(i));
				arg_remap[loop] = index->node_;
				CopyNodes(nodes_to_copy, arg_remap, {}, {}, false);
			}
		});

		//mark the loop for removal
		loops_to_remove.insert(loop);
	}

	// remove all loops that are not used
	for (auto* loop : loops_to_remove) {
		RemoveNode(loop);
	}

	UpdateGraph();
}


#define MAX_KERNEL_COPY_COST 2048.0f
void IR::OptimizeKernels() {
	// get kernel data
	vector<Node*> kernels = GetNodesOfType("kernel");
	ComputeNodeCost();

	// go over each kernel and copy computations outside the kernel if they are
	// cheap enough
	for (auto kernel : kernels) {
		ArgEdges args_to_copy;
		ArgEdges shape_args_to_copy;
		// go over all nodes in the kernel and check if their inputs can be copied
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			// go over all inputs
			for (auto& [arg, from]: node->args.inputs_) {
				bool inside_kernel = from->HasParent(kernel);
				bool from_in_kernel = from->HasParent("kernel");

				if (!inside_kernel && !node->args.CannotCopyArgument(arg))
				{
					// check if input is cheap enough to copy
					float input_cost = from->cost_;
					if (input_cost == -1.0) {
						//throw std::runtime_error("Cost has not been computed for node " + input.from_->get()->var_name);
						continue;
					}
					bool cheap_enough = input_cost >= 0.0f && input_cost < MAX_KERNEL_COPY_COST;
					bool has_only_one_output = from->args.outputs_.size() == 1;
					if (cheap_enough || has_only_one_output) {
						args_to_copy.push_back(ArgEdge(Arg(arg, from), *node));
					}
				}
				//shape arguments can not be inside kernels
				if (from_in_kernel && arg.first == ArgType::Shape) {
					shape_args_to_copy.push_back(ArgEdge(Arg(arg, from), *node));
				}
			}
		}

		//go over kernel shape arguments
		for (int i = 0; i < kernel->args.Count(ArgType::Shape); i++) {
			Node* shape_node = kernel->args.Get(ArgType::Shape, i);
			bool from_in_kernel =shape_node->HasParent("kernel");
			if (from_in_kernel) {
				shape_args_to_copy.push_back(ArgEdge(Arg(ArgID(ArgType::Shape, i), shape_node), kernel));
			}
		}

		// copy the nodes that are outside the kernel inside
		CopyArguments(args_to_copy, kernel->child);
		// copy shape arguments before the kernel
		CopyArguments(shape_args_to_copy, kernel);
	}
}

#define MAX_LOAD_COPY 5000.0f
#define MAX_LOAD_COPY_COUNT 2
#define MAX_LOAD_SIZE_RATIO 0.5f
void IR::OptimizeKernelLoadOperations() {
	ComputeNodeCost();

	vector<Node*> kernels = GetNodesOfType("kernel");

	unordered_set<Node*> nodes_to_remove;

	for (auto kernel : kernels) {
		ShapeInfo kernel_shape = ShapeInfo(kernel);

		unordered_map<Node*, Node*> loads_to_copy;
		unordered_set<Node*> memory_inputs;
		// go over all nodes in the kernel and check if their inputs can be copied
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			if (node->name != "load") continue;

			//get memory input
			Node* memory_input = node->args.Get(ArgType::Memory);

			//if(memory_input->debug_index == 423) {
			//	__debugbreak();
			//}

			ShapeInfo memory_shape = ShapeInfo(memory_input);

			bool inside_kernel = memory_input->HasParent("kernel");
			if (!inside_kernel) continue;

			bool is_not_modified = !memory_input->flags.has(NodeProp::Modified);
			if (!is_not_modified) continue;

			float kernel_size = ShapeInfo::GetSizeEstimate(kernel_shape);
			float memory_size = ShapeInfo::GetSizeEstimate(memory_shape);
			float size_ratio = kernel_size / memory_size;

			int output_count = (int)memory_input->args.outputs_.size();
			//only fuse if this is used less than MAX_LOAD_COPY_COUNT times or we can reduce dimensionality by fusing
			bool fusion_makes_sense = (output_count < MAX_LOAD_COPY_COUNT) ||
			                          (size_ratio <= MAX_LOAD_SIZE_RATIO) || memory_size == 1.0f;
			bool cheap_enough = memory_input->cost_ >= 0.0f &&
			                    memory_input->cost_ < (MAX_LOAD_COPY / output_count);


			//if the memory input is used only once and is not a memory node
			if (cheap_enough && fusion_makes_sense) {
				loads_to_copy[memory_input] = *node;
				memory_inputs.insert(memory_input);
			}
		}

		for (auto load : loads_to_copy) {
			//get the load
			Node* memory_input = load.first;
			Node* load_node = load.second;

			//get the indices
			unordered_map<int, Node*> indices;
			for (auto& [arg, from] : load_node->args.inputs_) {
				if (arg.first == ArgType::Index) {
					indices[arg.second] = from;
				}
			}

			//copy the load node
			map<Node*, Node*> copied_node_map = CopyNodesWithIndex({ memory_input }, indices, load_node);


			Tensors indices_tensors = Tensors();
			indices_tensors.resize(indices.size());
			for (auto& [index, node] : indices) {
				indices_tensors[index] = node->GetTensor();
			}

			//go over all the copied nodes and add load nodes to their inputs that are outside the kernel
			for (auto& [old_node, new_node] : copied_node_map) {
				AddNodeLoadOperations(new_node, kernel, indices_tensors);
			}

			Node* copied_load = copied_node_map[memory_input];
			//copy over the information from the original load node
			copied_load->CopyMetadata(load_node);

			//go over all outputs of the load node and replace them with the copied nodes
			for (auto [edge, to] : load_node->args.outputs_) {
				auto& [id, from] = edge;
				to->args.UpdateArgument(id, copied_load);
			}

			//remove the load node since it is not needed anymore
			nodes_to_remove.insert(load_node);
		}
	}

	// remove the load nodes
	for (auto node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();
}



#define MAX_HOST_COPY_COST 8192.0f

void IR::OptimizeHost() {
	ComputeNodeCost();

	//loop over all nodes and copy their arguments if they are cheap enough and inside kernels
	for (auto node = begin(); !node.end(); node.next()) {
		if (node->HasParent("kernel")) {
			continue;
		}

		ArgEdges args_to_copy;
		// go over all inputs
		for (auto& [arg, from] : node->args.inputs_) {
			bool inside_kernel = from->HasParent("kernel");

			if (inside_kernel && !node->args.CannotCopyArgument(arg)) {
				// check if input is cheap enough to copy
				float input_cost = from->cost_;
				if (input_cost == -1.0) {
					//throw std::runtime_error("Cost has not been computed for node " + input.from_->get()->var_name);
					continue;
				}
				bool cheap_enough = input_cost >= 0.0f && input_cost < MAX_HOST_COPY_COST;
				bool has_only_one_output = from->args.outputs_.size() == 1;

				if (cheap_enough || has_only_one_output) {
					args_to_copy.push_back(ArgEdge(Arg(arg, from), *node));
				} else {
					throw std::runtime_error("Host optimization: Copy cost too high for node " + node->name + " with cost " + to_string(input_cost));
				}
			}
		}

		CopyArguments(args_to_copy, node.get());
	}
}

} // namespace TensorFrost