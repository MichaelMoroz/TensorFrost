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

		bool has_other_loops = loop->HasChild("loop") || loop->HasParent("loop");

		if (!is_const || has_other_loops) {
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


void IR::UnrollKernelDimensions() {
	vector<Node*> kernels = GetNodesOfType("kernel");

	vector<Node*> nodes_to_remove;
	vector<pair<ArgEdges, Node*>> nodes_to_copy;
	for (auto kernel : kernels) {
		std::set<int> unused_dimensions;
		int dim = kernel->args.Count(ArgType::Shape);
		for (int i = 0; i < dim; i++) {
			unused_dimensions.insert(i);
		}

		bool can_unroll = true;
		bool has_atomics = false;
		for (auto node = NodeIterator(kernel); !node.end(); node.next()) {
			//if there are any nodes that have children, then we can not unroll
			if (node->child->valid()) {
				can_unroll = false;
				break;
			}

			//get all atomic scatter nodes
			if (node->op->HasAllTypes(OpProp::Scatter, OpProp::MemoryOp, OpProp::Modifier)) {
				has_atomics = true;
				//get the indices of the scatter operation
				NodeArguments indices = node->args.GetArguments(ArgType::Index);
				//get dependencies of all indices
				unordered_set<Node*> index_nodes;
				for (auto& [id, index] : indices) {
					index_nodes.insert(index);
				}
				unordered_set<Node*> dependencies = GetDependencies(index_nodes);
				//if any of the dependencies are a dim_id node, then its dimension(data[0]) is used
				for (auto dep : dependencies) {
					if (dep->name == "dim_id") {
						unused_dimensions.erase(dep->data[0]);
					}
				}
			}
		}

		int unused_count = (int)unused_dimensions.size();

		if (!can_unroll || unused_count == 0 || !has_atomics) {
			continue;
		}

		////modify the kernel to remove the unused dimensions

		//get smallest unused dimension
#define MAX_UNROLL_DIMENSIONS 2
		if(unused_count > MAX_UNROLL_DIMENSIONS) {
			unused_dimensions.erase(*unused_dimensions.begin());
		}


		//create new shape arguments
		auto old_shape = kernel->args.GetTensors(ArgType::Shape);
		Tensors new_shape;
		map<int, int> old_to_new;
		for (int i = 0; i < dim; i++) {
			if (!unused_dimensions.contains(i)) {
				new_shape.push_back(old_shape[i]);
				old_to_new[i] = (int)new_shape.size() - 1;
			}
		}

		//create new kernel
		Tensor* new_kernel = nullptr;
		ExecuteExpressionAfter(kernel, [&]() {
			new_kernel = &Tensor::Kernel(new_shape);
		});

		//create new dim_id nodes for the new kernel
		unordered_map<int, Node*> old_dim_to_node;

		ExecuteExpressionLastChild(new_kernel->node_, [&]() {
			for (int i = 0; i < dim; i++) {
				if (!unused_dimensions.contains(i)) {
					old_dim_to_node[i] = Tensor::Index(new_shape, old_to_new[i]).node_;
				}
			}
		});

		//add a loop for each unused dimension
		Tensor* last_loop = nullptr;
		Tensor* first_loop = nullptr;
		for (int dim : unused_dimensions) {
			const Tensor* shape = kernel->args.Get(ArgType::Shape, dim)->GetTensor();
			Node* parent = new_kernel->node_;
			if(last_loop != nullptr) {
				parent = last_loop->node_;
			}
			ExecuteExpressionLastChild(parent, [&]() {
				last_loop = &Tensor::Loop(Tensor::Constant(0), *shape, Tensor::Constant(1));
				last_loop->SetDebugName("dim_" + to_string(dim));
				old_dim_to_node[dim] = last_loop->node_;
				if(first_loop == nullptr) {
					first_loop = last_loop;
				}
			});
		}

		//move all old children to the last loop
		vector<Node*> old_kernel_nodes = GetChildren(kernel);
		vector<Node*> atomics_to_replace;
		last_loop->node_->child = kernel->child;
		for (auto node : old_kernel_nodes) {
			node->parent = last_loop->node_;
			//update shape arguments
			node->args.RemoveArguments(ArgType::Shape);
			for(int i = 0; i < new_shape.size(); i++) {
				node->args.AddArgument(ArgType::Shape, i, new_shape[i]->node_);
			}
			if (node->op->HasAllTypes(OpProp::Scatter, OpProp::MemoryOp, OpProp::Modifier)) {
				atomics_to_replace.push_back(node);
			}
		}

		kernel->child = nullptr;
		nodes_to_remove.push_back(kernel);

		UpdateGraph();

		//replace all old dim_id nodes with the loop index / new dim_id nodes
		for (auto node = NodeIterator(last_loop->node_); !node.end(); node.next()) {
			if (node->name == "dim_id") {
				int dim = node->data[0];
				if(old_dim_to_node.contains(dim)) {
					node->MakeOutputsUseGivenNode(old_dim_to_node[dim]);
				} else {
					throw std::runtime_error("Could not find new dim_id node for dimension " + to_string(dim) + " when optimizing kernel by unrolling dimensions");
				}
				nodes_to_remove.push_back(node.get());
			}
		}

		//create temporary accumulator nodes for the scatter operations
		unordered_map<Node*, Tensor*> scatter_to_accumulator;
		ExecuteExpressionFirstChild(new_kernel->node_, [&]() {
			for (auto node : atomics_to_replace) {
				Node* scatter_memory = node->args.Get(ArgType::Memory);
				if(node->name == "InterlockedAdd") {
					scatter_to_accumulator[node] = &Tensor::Constant(0, scatter_memory->type);
				} else if (node->name == "InterlockedMin") {
					scatter_to_accumulator[node] = &Tensor::Constant(GetInitialMin(node->type), scatter_memory->type);
				} else if (node->name == "InterlockedMax") {
					scatter_to_accumulator[node] = &Tensor::Constant(GetInitialMax(node->type), scatter_memory->type);
				} else if (node->name == "InterlockedAnd") {
					scatter_to_accumulator[node] = &Tensor::Constant(0xFFFFFFFF, scatter_memory->type);
				} else if (node->name == "InterlockedOr") {
					scatter_to_accumulator[node] = &Tensor::Constant(0, scatter_memory->type);
				} else if (node->name == "InterlockedXor") {
					scatter_to_accumulator[node] = &Tensor::Constant(0, scatter_memory->type);
				} else {
					scatter_to_accumulator[node] = &Tensor::Constant(0, scatter_memory->type);
				}
			}
		});

		//replace all scatter operations with the temporary accumulators
		for (auto& [scatter, accumulator] : scatter_to_accumulator) {
			//get the value to scatter
			const Tensor* value = scatter->args.Get(ArgType::Input, 0)->GetTensor();
			Tensor* accumulator_value = scatter_to_accumulator[scatter];
			ExecuteExpressionAfter(scatter, [&]() {
				if(scatter->name == "InterlockedAdd") {
					accumulator_value->Set(*accumulator + *value);
				} else if (scatter->name == "InterlockedMin") {
					accumulator_value->Set(Tensor::min(*accumulator, *value));
				} else if (scatter->name == "InterlockedMax") {
					accumulator_value->Set(Tensor::max(*accumulator, *value));
				} else if (scatter->name == "InterlockedAnd") {
					accumulator_value->Set(*accumulator & *value);
				} else if (scatter->name == "InterlockedOr") {
					accumulator_value->Set(*accumulator | *value);
				} else if (scatter->name == "InterlockedXor") {
					accumulator_value->Set(*accumulator ^ *value);
				} else {
					throw std::runtime_error("Unknown scatter operation " + scatter->name);
				}
			});
			nodes_to_remove.push_back(scatter);
		}


		//accumulate the temporary accumulators into the actual memory
		ArgEdges args_to_copy;
		ExecuteExpressionLastChild(new_kernel->node_, [&]() {
			for (auto& [scatter, accumulator] : scatter_to_accumulator) {
				//get the memory to scatter to
				const Tensor* memory = scatter->args.Get(ArgType::Memory)->GetTensor();
				Tensors indices = scatter->args.GetTensorVector(ArgType::Index);
				Tensor* accumulator_value = scatter_to_accumulator[scatter];
				Tensor* store_op = nullptr;
				Tensor* old_value = nullptr;
				if(unused_count <= MAX_UNROLL_DIMENSIONS) { //replace with a normal load and store operation
					old_value = &Tensor::Load(*memory, indices);
					Tensor* new_value = nullptr;
					if(scatter->name == "InterlockedAdd") {
						new_value = &(*old_value + *accumulator_value);
					} else if (scatter->name == "InterlockedMin") {
						new_value = &Tensor::min(*old_value, *accumulator_value);
					} else if (scatter->name == "InterlockedMax") {
						new_value = &Tensor::max(*old_value, *accumulator_value);
					} else if (scatter->name == "InterlockedAnd") {
						new_value = &(*old_value & *accumulator_value);
					} else if (scatter->name == "InterlockedOr") {
						new_value = &(*old_value | *accumulator_value);
					} else if (scatter->name == "InterlockedXor") {
						new_value = &(*old_value ^ *accumulator_value);
					} else {
						throw std::runtime_error("Unknown scatter operation " + scatter->name);
					}
					store_op = &Tensor::Store(*memory, *new_value, indices);
				} else { //still use atomic operations
					store_op = &Tensor::MemoryOp(scatter->name, memory, indices, accumulator_value);
				}
				NodeArguments store_indices = store_op->node_->args.GetArguments(ArgType::Index);
				for (auto& [id, from] : store_indices) {
					if (unused_count <= MAX_UNROLL_DIMENSIONS) {
						args_to_copy.push_back(ArgEdge(Arg(id, from), old_value->node_));
						args_to_copy.push_back(ArgEdge(Arg(id, from), store_op->node_));
					} else {
						args_to_copy.push_back(ArgEdge(Arg(id, from), store_op->node_));
					}
				}
			}
		});
		nodes_to_copy.push_back({args_to_copy, first_loop->node_->next});
	}

	for (auto node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();

	for (auto& [args, to] : nodes_to_copy) {
		CopyArguments(args, to);
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

#define MAX_LOAD_COPY 500000.0f
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