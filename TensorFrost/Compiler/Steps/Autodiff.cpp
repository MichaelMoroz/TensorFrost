#include "Compiler/KernelGen.h"

namespace TensorFrost {

const Tensor& ReduceGradientToShape(const Tensor& gradient, const Tensor& target) {
	ShapeCompareResult shape_result = CompareShape(gradient.node_, target.node_);
	if (!shape_result.compatible) {
		throw std::runtime_error("Autodiff: gradient shape not compatible with target tensor");
	}

	if(!shape_result.broadcast) {
		return gradient;
	}

	int dim = shape_result.broadcast_dim;
	ShapeInfo gradinfo = gradient.GetShapeInfo();
	ShapeInfo targetinfo = target.GetShapeInfo();

	gradinfo.ExpandDimensions(dim);
	targetinfo.ExpandDimensions(dim);

	Tensors grad_shape = gradinfo.GetTensors();
	Tensors target_shape = targetinfo.GetTensors();

	vector<int> axes_to_reduce;
	vector<bool> unsqueeze;
	for(int i = 0; i < dim; i++) {
		int val_a = grad_shape[i]->TryGetConstant();
		int val_b = target_shape[i]->TryGetConstant();
		if(val_a != val_b && val_b == 1) { //if the target has a dimension of 1, and the gradient has a different dimension, then reduce
			axes_to_reduce.push_back(i);
			bool should_unsqueeze = i >= (dim - target.GetDimension());
			unsqueeze.push_back(should_unsqueeze);
		}
	}

	Tensor* reduced = const_cast<Tensor*>(&gradient);
	//go in inverse order to keep the dimensions in the same order
	for(int i = (int)axes_to_reduce.size() - 1; i >= 0; i--) {
		reduced = &Tensor::Sum(*reduced, axes_to_reduce[i]);
		if(unsqueeze[i]) {
			reduced = &Tensor::Unsqueeze(*reduced, axes_to_reduce[i]);
		}
	}

	return *reduced;
}

class NodeGrads
{
	unordered_map<ArgID, Tensor*, HashArgID> argument_gradients;
	unordered_map<ArgID, const Tensor*, HashArgID> argument_inputs;
public:
	//get element at index
	const Tensor& operator[](ArgID id) {
		return *argument_gradients[id];
	}

	bool Contains(ArgID id) {
		return argument_gradients.contains(id);
	}

	bool Contains(ArgType type, int index = 0) {
		return Contains(ArgID(type, index));
	}

	NodeGrads(Node* node, map<Node*, Tensor*> input_grads) {
		for(auto& [id, input] : node->args.inputs_) {
			argument_inputs[id] = input->GetTensor();
			if(input_grads.contains(input)) {
				argument_gradients[id] = input_grads[input];
			}
		}
	}

	void Add(ArgType type, int index, Tensor& tensor) {
		const Tensor* target = argument_inputs[ArgID(type, index)];
		Tensor& new_tensor = const_cast<Tensor&>(ReduceGradientToShape(tensor, *target));
		if(Contains(type, index)) {
			argument_gradients[ArgID(type, index)] = &(*argument_gradients[ArgID(type, index)] + new_tensor);
		} else {
			argument_gradients[ArgID(type, index)] = &new_tensor;
		}
	}

	Tensor* GetGrad(ArgID id) {
		if(Contains(id)) {
			return argument_gradients[id];
		} else {
			Tensor* zero_grad = &Tensor::Constant(argument_inputs[id]->GetShape(), 0.0f);
			argument_gradients[id] = zero_grad;
			return zero_grad;
		}
	}

	Tensor* GetGrad(ArgType type, int index) {
		return GetGrad(ArgID(type, index));
	}

	//add gradients to inputs
	template <typename... Args>
	void Add(Tensor& arg, Args&... args) {
		//by default these are ArgType::Input
		vector<Tensor*> inputs = vector<Tensor*>({ &arg, &args... });
		for (int i = 0; i < inputs.size(); i++) {
			Add(ArgType::Input, i, *inputs[i]);
		}
	}
};

map<string, function<void(ArgumentManager&, Tensor&, Tensor&, NodeGrads&)>> gradient_functions =
{
	//elementwise operations
    {"copy", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad); }},
	{"add", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad, grad); }},
	{"sub", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad, -grad); }},
	{"mul", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * in[1], grad * in[0]); }},
	{"div", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad / in[1], -grad * in[0] / (in[1] * in[1])); }},
	{"neg", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(-grad); }},
	{"exp", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * out); }},
	{"log", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad / in[0]); }},
	{"sin", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * Tensor::cos(in[0])); }},
	{"cos", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(-grad * Tensor::sin(in[0])); }},
	{"tan", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * (Tensor::Constant(1.0f) + out * out)); }},
	{"asin", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad / Tensor::sqrt(Tensor::Constant(1.0f) - out * out)); }},
	{"acos", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(-grad / Tensor::sqrt(Tensor::Constant(1.0f) - out * out)); }},
	{"atan", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad / (Tensor::Constant(1.0f) + out * out)); }},
	{"abs", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * Tensor::sign(in[0])); }},
	{"sign", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f)); }},
	{"exp2", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * Tensor::log(Tensor::Constant(2.0f)) * out); }},
	{"log2", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad / (in[0] * Tensor::log(Tensor::Constant(2.0f)))); }},
	{"sqrt", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad / (Tensor::Constant(2.0f) * out)); }},
	{"rsqrt", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(-grad / (Tensor::Constant(2.0f) * in[0] * out)); }},
	{"floor", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f)); }},
	{"ceil", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f)); }},
	{"round", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f)); }},
	{"frac", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f)); }},
	{"atan2", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * in[1] / (in[0] * in[0] + in[1] * in[1]), -grad * in[0] / (in[0] * in[0] + in[1] * in[1])); }},
	{"lerp", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * in[2], grad * (Tensor::Constant(1.0f) - in[2]), grad * (in[0] - in[1])); }},
	{"max", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::select(in[0] > in[1], grad, Tensor::Constant(0.0f)), Tensor::select(in[0] < in[1], grad, Tensor::Constant(0.0f))); }},
	{"min", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::select(in[0] < in[1], grad, Tensor::Constant(0.0f)), Tensor::select(in[0] > in[1], grad, Tensor::Constant(0.0f))); }},
	{"pow", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * in[1] * Tensor::pow(in[0], in[1] - Tensor::Constant(1.0f)), grad * Tensor::log(in[0]) * out); }},
	{"tanh", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * (Tensor::Constant(1.0f) - out * out)); }},
	{"clamp", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		//clamp = min(max(x, min), max)
		Tensor& dc_dx = Tensor::select((in[0] < in[1]) || (in[0] > in[2]), Tensor::Constant(0.0f), grad);
		Tensor& dc_dmin = Tensor::select(in[0] < in[1], grad, Tensor::Constant(0.0f));
		Tensor& dc_dmax = Tensor::select(in[0] > in[2], grad, Tensor::Constant(0.0f));
		grads.Add(dc_dx, dc_dmin, dc_dmax);
	}},
	{"ternary", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f), Tensor::select(in[0], grad, Tensor::Constant(0.0f)), Tensor::select(in[0], Tensor::Constant(0.0f), grad)); }},
	{"lerp", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad * in[2], grad * (Tensor::Constant(1.0f) - in[2]), grad * (in[0] - in[1])); }},
	{"smoothstep", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		//smoothstep equation:
		//t = (x - e0) / (e1 - e0)
		//tc = clamp(t, 0.0, 1.0);
		//r = tc * tc * (3 - 2 * tc);
		//derivative of smoothstep:
		//dr/dx = dr/dtc * dtc/dt * dt/dx
		//dr/dtc = 6 * tc * (tc - 1)
		//dtc/dt = select((t < e0) || (t > e1), 0.0, 1.0)
		//dt/dx = 1 / (e1 - e0)
		//dt/dedge0 = (x - e1) / (e1 - e0)^2
		//dt/dedge1 = (e0 - x) / (e1 - e0)^2
		const Tensor& e0 = in[0];
		const Tensor& e1 = in[1];
		const Tensor& x = in[2];
		const Tensor& t = (x - e0) / (e1 - e0);
		const Tensor& tc = Tensor::clamp(t, Tensor::Constant(0.0f), Tensor::Constant(1.0f));
		const Tensor& dr_dtc = Tensor::Constant(6.0f) * tc * (tc - Tensor::Constant(1.0f));
		const Tensor& dtc_dt = Tensor::select((t < e0) || (t > e1), Tensor::Constant(0.0f), Tensor::Constant(1.0f));
		const Tensor& grad_dt = grad * dr_dtc * dtc_dt;
		const Tensor& dt_dx = Tensor::Constant(1.0f) / (e1 - e0);
		const Tensor& dt_de0 = (x - e1) * (dt_dx * dt_dx);
		const Tensor& dt_de1 = (e0 - x) * (dt_dx * dt_dx);
		grads.Add( grad_dt * dt_de0, grad_dt * dt_de1, grad_dt * dt_dx);
	}},
	{"step", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f), Tensor::Constant(0.0f)); }},
	{"modf", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(grad, Tensor::Constant(0.0f)); }},
	{"fma", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) { grads.Add(in[1] * grad, in[0] * grad, grad); }},

	//matrix operations
	{"matmul", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Matmul(grad, Tensor::Transpose(in[1])), Tensor::Matmul(Tensor::Transpose(in[0]), grad));
	}},
	{"transpose", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Transpose(grad, out.node_->data[1], out.node_->data[0]));
	}},
	{"dot", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		Tensor& unsq_grad = Tensor::Unsqueeze(grad, out.node_->data[0]);
		grads.Add(unsq_grad * in[1], unsq_grad * in[0]);
	}},
	{"unsqueeze", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Sqeeze(grad, out.node_->data[0]));
	}},
	{"dim_sum", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Unsqueeze(grad, out.node_->data[0]));
	}},
	{"dim_mean", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		int axis = (int)out.node_->data[0];
		Tensors shape = in[0].GetShape();
		Tensor& dim_size = Tensor::tofloat(*shape[axis]);
		grads.Add(Tensor::Unsqueeze(grad, axis) / dim_size);
	}},
	{"dim_norm", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		int axis = (int)out.node_->data[0];
		int dim1 = out.GetDimension();
		int dim2 = grad.GetDimension();
		axis = out.GetDimension() - axis - 1;
		axis = std::max(dim1, dim2) - axis - 1;
		//TODO: store axis from the right instead of the left
		Tensor& unsq = Tensor::Unsqueeze(grad/out, axis);
		grads.Add(unsq * in[0]);
	}},
	{"dim_max", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		auto& out_unsq = Tensor::Unsqueeze(out, out.node_->data[0]);
		auto& grad_unsq = Tensor::Unsqueeze(grad, out.node_->data[0]);
		grads.Add(Tensor::select(in[0] == out_unsq, grad_unsq, Tensor::Constant(0.0f)));
	}},
	{"dim_min", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		auto& out_unsq = Tensor::Unsqueeze(out, out.node_->data[0]);
		auto& grad_unsq = Tensor::Unsqueeze(grad, out.node_->data[0]);
		grads.Add(Tensor::select(in[0] == out_unsq, grad_unsq, Tensor::Constant(0.0f)));
	}},
	{"dim_prefix_sum", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		//b_i = a_0 + ... + a_i
		//db_i/da_j = 1 if i >= j, 0 otherwise
		//dL/da_j = sum_i dL/db_i * db_i/da_j
		//dL/da_j = sum_i dL/db_i * (i >= j)
		//g_i == dL/db_i
		//dL/da_j = g_j + g_{j+1} + ... + g_n = g_n + g_{n-1} + ... + g_j
		//c_i == g_{n-i}
		//dL/da_j = c_0 + c_1 + ... + c_j = prefix_sum(c)_j
		grads.Add(Tensor::PrefixSum(Tensor::Reverse(grad, out.node_->data[0]), out.node_->data[0]));
	}},
	{"dim_reverse", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Reverse(grad, out.node_->data[0]));
	}},
	{"reshape", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		const Tensor* memory_input = in.GetTensor(ArgType::Memory);
		grads.Add(ArgType::Memory, 0, Tensor::Reshape(grad, memory_input->GetShape()));
	}},
	{"assert", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		const Tensor* memory_input = in.GetTensor(ArgType::Memory);
		grads.Add(ArgType::Memory, 0, Tensor::Assert(grad, memory_input->GetShape(), memory_input->GetType()));
	}},
	//memory operations
	{"load", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		//derivative of load is scatter gradient to the load memory addresses
		int index_count = in.Count(ArgType::Index);

		Tensors tensor_indices = Tensors();
		for (int i = 0; i < index_count; i++) {
			tensor_indices.push_back(in.GetTensor(ArgType::Index, i));
		}

		Tensor& curGrad = *grads.GetGrad(ArgType::Memory, 0);
		Tensor::ScatterAdd(curGrad, grad, tensor_indices);
	}},
	{"store", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		//derivative of store is load gradient at the store memory addresses
		const Tensor* memory_input = in.GetTensor(ArgType::Memory);
		int index_count = in.Count(ArgType::Index);

		Tensors tensor_indices = Tensors();
		for (int i = 0; i < index_count; i++) {
			tensor_indices.push_back(in.GetTensor(ArgType::Index, i));
		}

		Tensor& memory_grad = *grads.GetGrad(ArgType::Memory, 0);
		grads.Add(ArgType::Input, 0, Tensor::Load(memory_grad, tensor_indices));
	}},
	{"InterlockedAdd", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		//derivative of scatter_add is load gradient at the scatter memory addresses
		const Tensor* memory_input = in.GetTensor(ArgType::Memory);
		int index_count = in.Count(ArgType::Index);

		Tensors tensor_indices = Tensors();
		for (int i = 0; i < index_count; i++) {
			tensor_indices.push_back(in.GetTensor(ArgType::Index, i));
		}

		Tensor& memory_grad = *grads.GetGrad(ArgType::Memory, 0);
		grads.Add(ArgType::Input, 0, Tensor::Load(memory_grad, tensor_indices));
	}},
	{"set", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		//derivative of set is the gradient of the setted value to the input
		Tensor& memory_grad = *grads.GetGrad(ArgType::Memory, 0);
		grads.Add(ArgType::Input, 0, memory_grad);
	}},
	{"detached_grad", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
	}},
	{"passthrough_grad", [](ArgumentManager& in, Tensor& out, Tensor& grad, NodeGrads& grads) {
		grads.Add(grad);
	}},
};

void ComputeNodeGradients(Node* value, Tensor* grad, NodeGrads& grads)
{
	string op_name = value->name;
	//add input arguments
	if(value->flags.has(NodeProp::PassGrad)) {
		op_name = "passthrough_grad";
	}
	if(value->flags.has(NodeProp::DetachGrad)) {
		op_name = "detached_grad";
	}

	if (!gradient_functions.contains(op_name)) {
		throw std::runtime_error("Cannot compute gradient for operation " + op_name);
	}

	Tensor out = *value->tensor_;
	gradient_functions[op_name](value->args, out, *grad, grads);
}

void IR::ComputeAutodiff()
{
	vector<Node*> gradients = GetNodesOfType(OpProp::Gradient);

	if(gradients.empty()) {
		return;
	}

	set<Node*> loss_nodes;
	map<pair<Node*, Node*>, Node*> loss_wrt_grad;
	unordered_map<Node*, int> min_range; //index of earliest node required for the gradient, end of backpropagation

	for (auto gradient : gradients) {
		Node* loss = gradient->args.Get(ArgType::Input, 0);
		Node* wrt = gradient->args.Get(ArgType::Input, 1);
		Node* last_loss_version = loss->GetLastVersion(gradient);

		loss_nodes.insert(last_loss_version);
		min_range[last_loss_version] = std::min(min_range[last_loss_version], wrt->index_);
		loss_wrt_grad[{last_loss_version, wrt}] = gradient;
	}

	map<Node*, Node*> grad_to_computed_grad;
	for (auto loss : loss_nodes) {
		set<Node*> visited;
		map<Node*, Tensor*> node_to_grad;

		unordered_set<Node*> loss_deps = GetDependencies({loss});

		//get all differentiable nodes that can change the loss
		vector<Node*> queue;
		for (auto dep : loss_deps) {
			bool in_range = (dep->index_ <= loss->index_ && dep->index_ >= min_range[loss]);
			bool dep_is_accessible = dep->HasCommonParents(loss); //is it in scope of the loss
			if(in_range && !dep->op->HasAllTypes(OpProp::Nondiff) &&
			   dep_is_accessible && (dep->type == TFType::Float || dep->op->HasAllTypes(OpProp::Modifier))) {
				queue.push_back(dep);
			}
		}

		//sort the nodes by index in descending order (backpropagation)
		ranges::sort(queue.begin(), queue.end(), [](Node* a, Node* b) {
			return a->index_ > b->index_;
		});

		Node* loss_value = loss;
		if(loss->op->HasAllTypes(OpProp::Modifier)) {
			loss_value = loss->args.Get(ArgType::Memory);
		}

		ExecuteExpressionAfter(loss, [&]() {
			node_to_grad[loss_value] = &Tensor::Constant(1.0f);
			for(auto node : queue) {
				if(!node_to_grad.contains(node) && !node->op->HasAllTypes(OpProp::Modifier)) {
					continue;
				}

				NodeGrads grads = NodeGrads(node, node_to_grad);
				ComputeNodeGradients(node, node_to_grad[node], grads);

				//store the computed gradients
				for (auto& [id, input]: node->args.inputs_) {
					if(!grads.Contains(id)) {
						continue;
					}

					Tensor& new_grad = *grads.GetGrad(id);
					node_to_grad[input] = &new_grad;

					//TODO: maybe add a function to get temp names
					if(input->debug_name != "") {
						new_grad.SetDebugName("d" + loss_value->debug_name + "_d" + input->debug_name);
					} else if(input->var_name != "") {
						new_grad.SetDebugName("d" + loss_value->debug_name + "_d" + input->var_name);
					}
				}
			}
		});

		for (auto wrt_grad : loss_wrt_grad) {
			if (wrt_grad.first.first != loss) {
				continue;
			}

			Node* grad = wrt_grad.second;
			if(!node_to_grad.contains(wrt_grad.first.second)) {
				throw std::runtime_error("Gradient not computed for " + wrt_grad.first.second->var_name);
			}
			Node* computed_grad = node_to_grad[wrt_grad.first.second]->node_;
			grad_to_computed_grad[grad] = computed_grad;
		}

		UpdateGraph();
	}

	unordered_set<Node*> nodes_to_remove;
	//replace all gradients with computed gradients
	for (auto gradient : gradients) {
		Node* computed_grad = grad_to_computed_grad[gradient];

		//replace the node with the sum
		gradient->MakeOutputsUseGivenNode(computed_grad);

		//copy over all memory flags to the new node
		computed_grad->flags.copy_all_given(gradient->flags, {NodeProp::InputMemory, NodeProp::OutputMemory});

		if (gradient->debug_name != "") {
			computed_grad->debug_name = gradient->debug_name;
		}

		//mark the node for removal
		nodes_to_remove.insert(gradient);

		UpdateGraph();
	}

	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();
}

} // namespace TensorFrost
