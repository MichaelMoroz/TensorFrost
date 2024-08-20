#include "Compiler/Implementations.h"

namespace TensorFrost {

const Tensor& ReduceGradientToShape(const Tensor& gradient, const Tensor& target)
{
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

	gradinfo.ExpandDimensionsTo(dim);
	targetinfo.ExpandDimensionsTo(dim);

	vector<int> axes_to_reduce;
	vector<bool> unsqueeze;
	for(int i = 0; i < dim; i++) {
		int val_a = gradinfo.GetTensor(i)->TryGetConstant();
		int val_b = targetinfo.GetTensor(i)->TryGetConstant();
		bool b_expanded = targetinfo.IsExpanded(i);
		if(b_expanded || (val_a != val_b && val_b == 1)) {
			axes_to_reduce.push_back(i);
			bool should_unsqueeze = i < target.GetDimension();
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

#ifndef NDEBUG
	//check if the reduced shape is the same as the target shape
	ShapeCompareResult result = CompareShape(reduced->node_, target.node_);
	if(!result.compatible) {
		throw std::runtime_error("Gradient shape not compatible with target tensor, function ReduceGradientToShape failed");
	}
#endif

	return *reduced;
}

map<string, VJPGradientFunction> gradient_functions =
{
	//elementwise operations
    {"copy", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad); }},
	{"add", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad, grad); }},
	{"sub", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad, -grad); }},
	{"mul", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad * in[1], grad * in[0]); }},
	{"div", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad / in[1], -grad * in[0] / (in[1] * in[1])); }},
	{"neg", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(-grad); }},
	{"exp", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad * out); }},
	{"log", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad / in[0]); }},
	{"sin", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad * Tensor::cos(in[0])); }},
	{"cos", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(-grad * Tensor::sin(in[0])); }},
	{"tan", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad * (Tensor::Constant(1.0f) + out * out)); }},
	{"asin", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad / Tensor::sqrt(Tensor::Constant(1.0f) - out * out)); }},
	{"acos", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(-grad / Tensor::sqrt(Tensor::Constant(1.0f) - out * out)); }},
	{"atan", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad / (Tensor::Constant(1.0f) + out * out)); }},
	{"abs", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad * Tensor::sign(in[0])); }},
	{"sign", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f)); }},
	{"exp2", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad * Tensor::Constant(log(2.0f)) * out); }},
	{"log2", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad / (in[0] * Tensor::Constant(log(2.0f)))); }},
	{"sqrt", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad / (Tensor::Constant(2.0f) * out)); }},
	{"rsqrt", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(-grad / (Tensor::Constant(2.0f) * in[0] * out)); }},
	{"floor", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f)); }},
	{"ceil", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f)); }},
	{"round", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f)); }},
	{"frac", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f)); }},
	{"atan2", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad * in[1] / (in[0] * in[0] + in[1] * in[1]), -grad * in[0] / (in[0] * in[0] + in[1] * in[1])); }},
	{"lerp", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad * in[2], grad * (Tensor::Constant(1.0f) - in[2]), grad * (in[0] - in[1])); }},
	{"max", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::select(in[0] > in[1], grad, Tensor::Constant(0.0f)), Tensor::select(in[0] < in[1], grad, Tensor::Constant(0.0f))); }},
	{"min", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::select(in[0] < in[1], grad, Tensor::Constant(0.0f)), Tensor::select(in[0] > in[1], grad, Tensor::Constant(0.0f))); }},
	{"pow", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad * in[1] * Tensor::pow(in[0], in[1] - Tensor::Constant(1.0f)), grad * Tensor::log(in[0]) * out); }},
	{"tanh", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad * (Tensor::Constant(1.0f) - out * out)); }},
	{"clamp", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		//clamp = min(max(x, min), max)
		Tensor& dc_dx = Tensor::select((in[0] < in[1]) || (in[0] > in[2]), Tensor::Constant(0.0f), grad);
		Tensor& dc_dmin = Tensor::select(in[0] < in[1], grad, Tensor::Constant(0.0f));
		Tensor& dc_dmax = Tensor::select(in[0] > in[2], grad, Tensor::Constant(0.0f));
		grads.Add(dc_dx, dc_dmin, dc_dmax);
	}},
	{"ternary", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f), Tensor::select(in[0], grad, Tensor::Constant(0.0f)), Tensor::select(in[0], Tensor::Constant(0.0f), grad)); }},
	{"lerp", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad * in[2], grad * (Tensor::Constant(1.0f) - in[2]), grad * (in[0] - in[1])); }},
	{"smoothstep", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
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
	{"step", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(Tensor::Constant(0.0f), Tensor::Constant(0.0f)); }},
	{"modf", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(grad, Tensor::Constant(0.0f)); }},
	{"fma", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) { grads.Add(in[1] * grad, in[0] * grad, grad); }},

	//matrix operations
	{"matmul", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Matmul(grad, Tensor::Transpose(in[1])), Tensor::Matmul(Tensor::Transpose(in[0]), grad));
	}},
	{"transpose", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Transpose(grad, out.axis(1), out.axis(0)));
	}},
	{"dot", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		Tensor& unsq_grad = Tensor::Unsqueeze(grad, out.axis());
		grads.Add(unsq_grad * in[1], unsq_grad * in[0]);
	}},
	{"unsqueeze", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Squeeze(grad, out.axis()));
	}},
	{"squeeze", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Unsqueeze(grad, out.axis()));
	}},
	{"dim_sum", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Unsqueeze(grad, out.axis()));
	}},
	{"dim_mean", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		int axis = out.axis();
		Tensors shape = in[0].GetShape();
		Tensor& dim_size = Tensor::tofloat(*shape[axis]);
		grads.Add(Tensor::Unsqueeze(grad, axis) / dim_size);
	}},
	{"dim_norm", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		//TODO: store axis from the right instead of the left
		Tensor& unsq = Tensor::Unsqueeze(grad/out, out.axis());
		grads.Add(unsq * in[0]);
	}},
	{"dim_max", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		auto& out_unsq = Tensor::Unsqueeze(out, out.axis());
		auto& grad_unsq = Tensor::Unsqueeze(grad, out.axis());
		grads.Add(Tensor::select(in[0] == out_unsq, grad_unsq, Tensor::Constant(0.0f)));
	}},
	{"dim_min", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		auto& out_unsq = Tensor::Unsqueeze(out, out.axis());
		auto& grad_unsq = Tensor::Unsqueeze(grad, out.axis());
		grads.Add(Tensor::select(in[0] == out_unsq, grad_unsq, Tensor::Constant(0.0f)));
	}},
	{"dim_prefix_sum", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
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
	{"dim_reverse", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		grads.Add(Tensor::Reverse(grad, out.node_->data[0]));
	}},
	{"reshape", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		const Tensor* memory_input = in.GetTensor(ArgType::Memory);
		grads.Add(ArgType::Memory, 0, Tensor::Reshape(grad, memory_input->GetShape()));
	}},
	{"assert", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		const Tensor* memory_input = in.GetTensor(ArgType::Memory);
		grads.Add(ArgType::Memory, 0, Tensor::Assert(grad, memory_input->GetShape(), memory_input->GetType()));
	}},
	//memory operations
	{"load", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		//derivative of load is scatter gradient to the load memory addresses
		int index_count = in.Count(ArgType::Index);

		Tensors tensor_indices = Tensors();
		for (int i = 0; i < index_count; i++) {
			tensor_indices.push_back(in.GetTensor(ArgType::Index, i));
		}

		const Tensor& curGrad = *grads.GetGrad(ArgType::Memory, 0);
		Tensor::ScatterAdd(curGrad, grad, tensor_indices);
	}},
	{"store", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		//derivative of store is load gradient at the store memory addresses
		const Tensor* memory_input = in.GetTensor(ArgType::Memory);
		int index_count = in.Count(ArgType::Index);

		Tensors tensor_indices = Tensors();
		for (int i = 0; i < index_count; i++) {
			tensor_indices.push_back(in.GetTensor(ArgType::Index, i));
		}

		const Tensor& memory_grad = *grads.GetGrad(ArgType::Memory, 0);
		grads.Add(ArgType::Input, 0, Tensor::Load(memory_grad, tensor_indices));
	}},
	{"InterlockedAdd", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		//derivative of scatter_add is load gradient at the scatter memory addresses
		const Tensor* memory_input = in.GetTensor(ArgType::Memory);
		int index_count = in.Count(ArgType::Index);

		Tensors tensor_indices = Tensors();
		for (int i = 0; i < index_count; i++) {
			tensor_indices.push_back(in.GetTensor(ArgType::Index, i));
		}

		const Tensor& memory_grad = *grads.GetGrad(ArgType::Memory, 0);
		grads.Add(ArgType::Input, 0, Tensor::Load(memory_grad, tensor_indices));
	}},
	{"set", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		//derivative of set is the gradient of the setted value to the input
		const Tensor& memory_grad = *grads.GetGrad(ArgType::Memory, 0);
		grads.Add(ArgType::Input, 0, memory_grad);
	}},
	{"detached_grad", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
	}},
	{"passthrough_grad", [](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		grads.Add(grad);
	}},
};

VJPGradientFunction GetVJPForOperation(string name) {
	if (!gradient_functions.contains(name)) {
		throw std::runtime_error("Cannot compute VJP for operation " + name);
	}
	return gradient_functions[name];
}

void RegisterVJP(string name, VJPGradientFunction vjp) {
	if (gradient_functions.contains(name)) {
		throw std::runtime_error("VJP for operation " + name + " already registered");
	}
	gradient_functions[name] = vjp;
}

Tensor* ComputeReduction(const Tensor* array, int axis,
                         std::function<Tensor*(Tensor*, Tensor*)> reduction_op, string debug_name = "",
                         uint initial = 0,
                         std::function<Tensor*(Tensor*)> element_op = nullptr) {
	// Get shape of the array
	Tensors shape = array->GetShape();

	axis = GetAxis((int)shape.size(), axis);

	// Get the number of dimensions
	int dims = (int)shape.size();

	Tensors sum_shape = Tensors();
	for (int i = 0; i < dims; i++) {
		if (i == axis) {
			continue;
		}
		sum_shape.push_back(shape[i]);
	}

	// get indices for all dimensions but the last
	Tensors indices = Tensors();
	for (int i = 0; i < dims - 1; i++) {
		indices.push_back(&Tensor::Index(sum_shape, i));
	}

	// if no dimensions, then add constant 1
	if (sum_shape.empty()) {
		sum_shape.push_back(&Tensor::Constant(1));
	}

	Tensors load_index = Tensors();
	for (int id = 0, d = 0; d < dims; d++) {
		if (d == axis) {
			load_index.push_back(&Tensor::Constant(sum_shape, 0));
		} else {
			load_index.push_back(indices[id++]);
		}
	}

	// start with the first value
	Tensor* reduced = &Tensor::Constant(sum_shape, initial, array->node_->type);
	reduced->SetDebugName(debug_name);

	// create a loop over the last dimension starting from the second value
	Tensor::Loop(Tensor::Constant(0), *shape[axis], Tensor::Constant(1),
	[&](const Tensor& i) {
		load_index[axis] = &i;

		// load the value
		Tensor* value = &Tensor::Load(*array, load_index, IndexingMode::Unsafe);

		if (element_op != nullptr) {
			value = element_op(value);
		}

		reduced->Set(*reduction_op(reduced, value));
	});

	return reduced;
}

Tensor* ComputeScan(const Tensor* array, int axis, std::function<Tensor*(Tensor*, Tensor*)> scan_op, string debug_name = "", uint initial = 0) {
	// Get shape of the array
	Tensors shape = array->GetShape();

	Tensor* scan_result = &Tensor::Memory(shape, array->node_->type);

	axis = GetAxis((int)shape.size(), axis);

	// Get the number of dimensions
	int dims = (int)shape.size();

	Tensors sum_shape = Tensors();
	for (int i = 0; i < dims; i++) {
		if (i == axis) {
			continue;
		}
		sum_shape.push_back(shape[i]);
	}

	// get indices for all dimensions but the last
	Tensors indices = Tensors();
	for (int i = 0; i < dims - 1; i++) {
		indices.push_back(&Tensor::Index(sum_shape, i));
	}

	// if no dimensions, then add constant 1
	if (sum_shape.empty()) {
		sum_shape.push_back(&Tensor::Constant(1));
	}

	Tensors load_index = Tensors();
	for (int id = 0, d = 0; d < dims; d++) {
		if (d == axis) {
			load_index.push_back(&Tensor::Constant(sum_shape, 0));
		} else {
			load_index.push_back(indices[id++]);
		}
	}

	// start with the first value
	Tensor* reduced = &Tensor::Constant(sum_shape, initial, array->node_->type);
	reduced->SetDebugName(debug_name);

	// create a loop over the last dimension starting from the second value
	Tensor::Loop(Tensor::Constant(0), *shape[axis], Tensor::Constant(1),
	[&](const Tensor& i) {
		load_index[axis] = &i;
		// load the value
		Tensor* value = &Tensor::Load(*array, load_index, IndexingMode::Unsafe);
		reduced->Set(*scan_op(reduced, value));
		Tensor::Store(*scan_result, *reduced, load_index, true);
	});

	return scan_result;
}

Tensor* ComputeSum(const Tensor* array, int axis) {
	return ComputeReduction(array, axis, [](Tensor* a, Tensor* b) {
	return &(*a + *b); }, "sum");
}

Tensor* ComputeNorm(const Tensor* array, int axis) {
	return &Tensor::sqrt(Tensor::tofloat(*ComputeReduction(array, axis,
		[](Tensor* a, Tensor* b) { return &(*a + *b); }, "norm", 0,
		[](Tensor* a) { return &(*a * *a); })));
}

Tensor* ComputeMean(const Tensor* array, int axis) {
	Tensor* sum = ComputeSum(array, axis);
	Tensors shape = array->GetShape();
	axis = GetAxis((int)shape.size(), axis);
	return &(Tensor::tofloat(*sum) / Tensor::tofloat(*shape[axis]));
}

uint GetInitialMax(TFType type) {
	if (type == TFType::Float) {
		float init = -FLT_MAX;
		return *(uint*)&init;
	}
	else if (type == TFType::Int) {
		int init = INT_MIN;
		return *(uint*)&init;
	}
	return 0;
}

uint GetInitialMin(TFType type) {
	if (type == TFType::Float) {
		float init = FLT_MAX;
		return *(uint*)&init;
	}
	else if (type == TFType::Int) {
		int init = INT_MAX;
		return *(uint*)&init;
	}
	return 0;
}

Tensor* ComputeMax(const Tensor* array, int axis) {
	uint initial = 0;
	if (array->node_->type == TFType::Float) {
		float init = -FLT_MAX;
		initial = *(uint*)&init;
	}
	else if (array->node_->type == TFType::Int) {
		int init = INT_MIN;
		initial = *(uint*)&init;
	}
	return ComputeReduction(
	    array, axis, [](Tensor* a, Tensor* b) { return &Tensor::max(*a, *b); },
	    "max", initial);
}

Tensor* ComputeMin(const Tensor* array, int axis) {
	uint initial = UINT_MAX;
	if (array->node_->type == TFType::Float) {
		float init = FLT_MAX;
		initial = *(uint*)&init;
	}
	else if (array->node_->type == TFType::Int) {
		int init = INT_MAX;
		initial = *(uint*)&init;
	}
	return ComputeReduction(
	    array, axis, [](Tensor* a, Tensor* b) { return &Tensor::min(*a, *b); },
	    "min", initial);
}

Tensor* ComputeProduct(const Tensor* array, int axis) {
	uint initial = 1;
	if (array->node_->type == TFType::Float) {
		float init = 1.0f;
		initial = *(uint*)&init;
	}
	return ComputeReduction(
	    array, axis, [](Tensor* a, Tensor* b) { return &(*a * *b); }, "prod", initial);
}

Tensor* ComputeAny(const Tensor* array, int axis) {
	return ComputeReduction(array, axis, [](Tensor* a, Tensor* b) { return &(*a || *b); }, "any", 0);
}

Tensor* ComputeAll(const Tensor* array, int axis) {
	return ComputeReduction(
	    array, axis, [](Tensor* a, Tensor* b) { return &(*a && *b); }, "all", ~0);
}

Tensor* ComputePrefixSum(const Tensor* array, int axis) {
	return ComputeScan(array, axis, [](Tensor* a, Tensor* b) { return &(*a + *b); }, "prefix_sum");
}

Tensor* Transpose(const Tensor* array, map<int, int> permutation) {
	ShapeInfo shapeinfo = array->GetShapeInfo();
	int old_dim = shapeinfo.dim;
	Tensors perm_shape = Tensors();
	int permuted_dim = (int)permutation.size();

	shapeinfo.ExpandDimensionsTo(permuted_dim);
	Tensors shape = shapeinfo.GetTensors();

	for (int i = 0; i < permuted_dim; i++) {
		perm_shape.push_back(shape[permutation[i]]);
	}

	//create indices
	Tensors indices = Tensors();
	for (int i = 0; i < permuted_dim; i++) {
		indices.push_back(&Tensor::Index(perm_shape, i));
	}
	//permute indices to load the values
	Tensors perm_indices = Tensors(old_dim, nullptr);
	for (int i = 0; i < permuted_dim; i++) {
		int old = permutation[i];
		if(old < old_dim) {
			perm_indices[old] = indices[i];
		}
	}
	//if any nullptr, then put a constant 0
	for (int i = 0; i < old_dim; i++) {
		if(perm_indices[i] == nullptr) {
			perm_indices[i] = &Tensor::Constant(0);
		}
	}

	Tensor& loaded = Tensor::Load(*array, perm_indices, IndexingMode::Unsafe);
	loaded.SetDebugName("transposed");
	return &loaded;
}

Tensor* ReverseDim(const Tensor* array, int axis) {
	ShapeInfo shapeinfo = array->GetShapeInfo();
	int dims = shapeinfo.dim;
	Tensors shape = shapeinfo.GetTensors();
	Tensors indices = Tensors();
	for (int i = 0; i < dims; i++) {
		if (i == axis) {
			indices.push_back(&(*shape[i] - Tensor::Constant(1) - Tensor::Index(shape, i)));
		} else {
			indices.push_back(&Tensor::Index(shape, i));
		}
	}
	Tensor& loaded = Tensor::Load(*array, indices, IndexingMode::Unsafe);
	loaded.SetDebugName("reversed");
	return &loaded;
}

Tensor* ConstantOutOfBounds(const Tensor* array, Tensors indices, uint constant) {
	ShapeInfo shapeinfo = array->GetShapeInfo();
	int dims = shapeinfo.dim;
	Tensors shape = shapeinfo.GetTensors();
	Tensor* is_out_of_bounds = &Tensor::Constant(0, TFType::Bool);
	for (int i = 0; i < dims; i++) {
		Tensor* is_out = &(*indices[i] < Tensor::Constant(0) || *indices[i] >= *shape[i]);
		is_out_of_bounds = &(*is_out_of_bounds || *is_out);
	}
	is_out_of_bounds->SetDebugName("out_of_bounds");
	Tensor* value = &Tensor::Constant(constant, array->node_->type);
	Tensor* loaded = &Tensor::Load(*array, indices);
	return &Tensor::select(*is_out_of_bounds, *value, *loaded);
}

Tensor* SplitDim(const Tensor* array, const Tensor* splitted, int axis, int split_size) {
	ShapeInfo shapeinfo = array->GetShapeInfo();
	int dims = shapeinfo.dim;
	Tensors new_shape = splitted->GetShape();
	Tensors indices = Tensors();
	for (int i = 0; i < dims; i++) {
		if (i == axis) {
			Tensor* index1 = &Tensor::Index(new_shape, i);
			Tensor* index2 = &Tensor::Index(new_shape, i + 1);
			//merged index
			indices.push_back(&(*index2 * (*new_shape[i]) + *index1));
		} else if(i < axis) {
			indices.push_back(&Tensor::Index(new_shape, i));
		} else {
			indices.push_back(&Tensor::Index(new_shape, i + 1));
		}
	}
	Tensor* loaded = ConstantOutOfBounds(array, indices, 0);
	loaded->SetDebugName("split");
	return loaded;
}

Tensor* MergeDim(const Tensor* array, const Tensor* merged, int axis) {
	ShapeInfo shapeinfo = array->GetShapeInfo();
	int dims = shapeinfo.dim;
	axis = GetAxis(dims, axis);
	Tensors shape = shapeinfo.GetTensors();
	Tensors new_shape = merged->GetShape();
	Tensors indices = Tensors();
	for (int i = 0; i < dims-1; i++) {
		if (i == axis) {
			Tensor* merged_index = &Tensor::Index(new_shape, i);
			//get split index
			indices.push_back(&(*merged_index % *shape[axis]));
			indices.push_back(&(*merged_index / *shape[axis]));
		} else {
			indices.push_back(&Tensor::Index(new_shape, i));
		}
	}
	Tensor* loaded = ConstantOutOfBounds(array, indices, 0);
	loaded->SetDebugName("merged");
	return loaded;
}

Tensor* ComputeDot(const Tensor* a, const Tensor* b, int axis) {
	Tensors shape_a = a->GetShape();
	Tensors shape_b = b->GetShape();
	axis = GetAxis((int)shape_a.size(), axis);
	return ComputeSum(&(*a * *b), axis);
}

//compute the matrix multiplication of two last dimensions
//takes two tensors [T1, T2, ..., Tn, M, N] and [Tm, .., Tn, N, K] and returns [T1, T2, ..., Tm, M, K]
Tensor* ComputeMatMul(const Tensor* a, const Tensor* b) {
	ShapeInfo shape_a = a->GetShapeInfo();
	ShapeInfo shape_b = b->GetShapeInfo();

	if (shape_a.dim < 2 && shape_b.dim < 2) {
		throw std::runtime_error("Matrix multiplication requires at least one 2D tensor");
	}

	if(shape_a.dim < 2) {
		shape_a.ExpandDimensionsTo(2);
	}
	if(shape_b.dim < 2) {
		shape_b.ExpandDimensionsTo(2);
	}

	Tensors shape_a_tensors = shape_a.GetTensors();
	Tensors shape_b_tensors = shape_b.GetTensors();

	//get shape of the result
	Tensors shape_c = Tensors();
	int dim_a = shape_a.dim;
	int dim_b = shape_b.dim;
	int max_dim = 0;
	Tensors max_shape = Tensors();
	//get the shape with most dimensions
	if (dim_a < dim_b) {
		max_dim = dim_b;
		max_shape = shape_b_tensors;
	} else {
		max_dim = dim_a;
		max_shape = shape_a_tensors;
	}

	shape_c.push_back(shape_b_tensors[0]);
	shape_c.push_back(shape_a_tensors[1]);
	for (int i = 2; i < max_dim; i++) {
		shape_c.push_back(max_shape[i]);
	}
	ShapeDimCompareResult result = CompareShapeDim(shape_a_tensors[0]->node_, shape_b_tensors[1]->node_);
	if (!result.compatible) {
		throw std::runtime_error("Inner dimensions of the matrices must match");
	}

	const Tensor* sum_shape = result.broadcast_dim->GetTensor();

	// get indices for c elements
	Tensors indices_c = Tensors();
	for (int i = 0; i < max_dim; i++) {
		indices_c.push_back(&Tensor::Index(shape_c, i));
	}

	// start with 0
	Tensor* c = &Tensor::Constant(shape_c, 0, a->node_->type);
	c->SetDebugName("matmul");

	// loop over k and compute += A t1t2..tN ik * B t1t2..tN kj
	Tensor::Loop(Tensor::Constant(0), *sum_shape, Tensor::Constant(1),
		[&](const Tensor& k) {

		// get indices for a elements
		Tensors indices_a = Tensors();

		indices_a.push_back(&k);
		indices_a.push_back(indices_c[1]);
		for (int i = 2; i < dim_a; i++) {
			indices_a.push_back(indices_c[i]);
		}

		// get indices for b elements
		Tensors indices_b = Tensors();

		indices_b.push_back(indices_c[0]);
		indices_b.push_back(&k);
		for (int i = 2; i < dim_b; i++) {
			indices_b.push_back(indices_c[i]);
		}

		// load the value
		Tensor* value = &(Tensor::Load(*a, indices_a, IndexingMode::Unsafe) *
		                  Tensor::Load(*b, indices_b, IndexingMode::Unsafe));

		c->Set(*c + *value);
	});

	return c;
}

map<string, ImplementationFunction> implementation_functions =
{
	{"dim_sum", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) { outputs.push_back(ComputeSum(inputs[0],axes[0])); }},
	{"dim_norm", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) { outputs.push_back(ComputeNorm(inputs[0],axes[0])); }},
	{"dim_max", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) { outputs.push_back(ComputeMax(inputs[0],axes[0])); }},
	{"dim_min", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) { outputs.push_back(ComputeMin(inputs[0],axes[0])); }},
	{"dim_mean", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) { outputs.push_back(ComputeMean(inputs[0],axes[0])); }},
	{"dim_product", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) { outputs.push_back(ComputeProduct(inputs[0],axes[0])); }},
	{"dim_any", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) { outputs.push_back(ComputeAny(inputs[0],axes[0])); }},
	{"dim_all", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) { outputs.push_back(ComputeAll(inputs[0],axes[0])); }},
	{"dim_prefix_sum", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) { outputs.push_back(ComputePrefixSum(inputs[0],axes[0])); }},
	{"transpose", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) {
		//get the permutation
		int dim = (int)inputs[0]->GetDimension();
		dim = std::max(dim, std::max(axes[0], axes[1]) + 1);
		map<int, int> permutation;
		for (int i = 0; i < dim; i++) {
			if(i == axes[0]) {
				permutation[i] = axes[1];
			} else if(i == axes[1]) {
				permutation[i] = axes[0];
			} else {
				permutation[i] = i;
			}
		}
		outputs.push_back(Transpose(inputs[0], permutation));
	}},
	{"dot", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) { outputs.push_back(ComputeDot(inputs[0], inputs[1], axes[0])); }},
	{"matmul", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) { outputs.push_back(ComputeMatMul(inputs[0], inputs[1])); }},
	{"unsqueeze", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) {
		map<int, int> permutation;
		int dim = (int)inputs[0]->GetDimension()+1;
		dim = std::max(dim, axes[0] + 1);
		for(int i = 0; i < dim; i++) {
			if(i == axes[0]) {
				permutation[i] = dim-1;
			} else if (i < axes[0]) {
				permutation[i] = i;
			} else {
				permutation[i] = i - 1;
			}
		}
		outputs.push_back(Transpose(inputs[0], permutation));
	}},
	{"squeeze", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) {
		map<int, int> permutation;
		int dim = (int)inputs[0]->GetDimension() - 1;
		for(int i = 0; i < dim; i++) {
			if(i < axes[0]) {
				permutation[i] = i;
			} else {
				permutation[i] = i + 1;
			}
		}
		outputs.push_back(Transpose(inputs[0], permutation));
	}},
	{"dim_reverse", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) { outputs.push_back(ReverseDim(inputs[0], axes[0])); }},
	{"dim_split", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) { outputs.push_back(SplitDim(inputs[0], tensor, axes[0], axes[1])); }},
	{"dim_merge", [](Tensors& outputs, map<int, const Tensor*> inputs, const Tensor* tensor,vector<int> axes ) { outputs.push_back(MergeDim(inputs[0], tensor, axes[0])); }},
};

ImplementationFunction GetImplementationForOperation(string name) {
	if (!implementation_functions.contains(name)) {
		throw std::runtime_error("Cannot compute implementation for operation " + name);
	}
	return implementation_functions[name];
}

void RegisterImplementation(string name, ImplementationFunction impl) {
	if (implementation_functions.contains(name)) {
		throw std::runtime_error("Implementation for operation " + name + " already exists");
	}
	implementation_functions[name] = impl;
}


map<string, AlgorithmVJPGradientFunction> algorithm_vjps = {};

AlgorithmVJPGradientFunction GetAlgorithmVJPForOperation(string name) {
	if (!algorithm_vjps.contains(name)) {
		throw std::runtime_error("Cannot compute VJP for operation " + name);
	}
	return algorithm_vjps[name];
}

void RegisterAlgorithmVJP(string name, AlgorithmVJPGradientFunction vjp) {
	if (algorithm_vjps.contains(name)) {
		throw std::runtime_error("VJP for operation " + name + " already registered");
	}
	algorithm_vjps[name] = vjp;
}

VJPGradientFunction CreateAlgorithmVJP(const string& name) {
	VJPGradientFunction vjp = [name](ArgumentManager& in, const Tensor& out, const Tensor& grad, NodeGrads& grads) {
		auto inputs = in.GetTensors(ArgType::Input);
		AlgorithmVJPGradientFunction impl = GetAlgorithmVJPForOperation(name);
		Tensors grad_tensors = impl(inputs, &grad, &out);
		for (int i = 0; i < (int)grad_tensors.size(); i++) {
			grads.Add(ArgType::Input, i, *const_cast<Tensor*>(grad_tensors[i]));
		}
	};
	return vjp;
}

void RegisterAlgorithmicPrimitive(const string& name, vector<string> overloads,  ImplementationFunction impl, AlgorithmVJPGradientFunction vjp) {
	Operation* newop = new Operation(name, overloads, 0, "", {OpProp::Custom, OpProp::Algorithm});
	RegisterNewOperation(newop);
	RegisterImplementation(name, impl);
	RegisterAlgorithmVJP(name, vjp);
	RegisterVJP(name, CreateAlgorithmVJP(name));
}

} // namespace TensorFrost


