#include "Compiler/KernelGen.h"

namespace TensorFrost {


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
		Tensor* value = &Tensor::Load(*array, load_index, true);

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
		Tensor* value = &Tensor::Load(*array, load_index, true);
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

	shapeinfo.ExpandDimensions(permuted_dim);
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
		int old = permutation[i] - std::max(permuted_dim - old_dim, 0);
		if(old >= 0) {
			perm_indices[old] = indices[i];
		}
	}
	//if any nullptr, then put a constant 0
	for (int i = 0; i < old_dim; i++) {
		if(perm_indices[i] == nullptr) {
			perm_indices[i] = &Tensor::Constant(0);
		}
	}

	Tensor& loaded = Tensor::Load(*array, perm_indices, true);
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
	Tensor& loaded = Tensor::Load(*array, indices, true);
	loaded.SetDebugName("reversed");
	return &loaded;
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
		shape_a.ExpandDimensions(2);
	}
	if(shape_b.dim < 2) {
		shape_b.ExpandDimensions(2);
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

	for (int i = 0; i < max_dim - 2; i++) {
		shape_c.push_back(max_shape[i]);
	}
	shape_c.push_back(shape_a_tensors[dim_a - 2]);
	shape_c.push_back(shape_b_tensors[dim_b - 1]);

	ShapeDimCompareResult result = CompareShapeDim(shape_a_tensors[dim_a - 1]->node_, shape_b_tensors[dim_b - 2]->node_);
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
		for (int i = 0; i < dim_a - 2; i++) {
			indices_a.push_back(indices_c[max_dim - dim_a + i]);
		}
		indices_a.push_back(indices_c[max_dim - 2]);
		indices_a.push_back(&k);

		// get indices for b elements
		Tensors indices_b = Tensors();
		for (int i = 0; i < dim_b - 2; i++) {
			indices_b.push_back(indices_c[max_dim - dim_b + i]);
		}
		indices_b.push_back(&k);
		indices_b.push_back(indices_c[max_dim - 1]);

		// load the value
		Tensor* value = &(Tensor::Load(*a, indices_a, true) *
		                  Tensor::Load(*b, indices_b, true));

		c->Set(*c + *value);
	});

	return c;
}

void IR::InsertAlgorithmicPrimitives() {
	// get all nodes for each type
	vector<Node*> nodes = GetNodesOfType(OpProp::Algorithm);

	unordered_set<Node*> nodes_to_remove;

	// replace all nodes with the algorithmic primitive
	for (auto node : nodes) {
		//compute the sum after the node
		ExecuteExpressionAfter(node, [&]() {
			//get the input tensor
			map<int, const Tensor*> inputs = node->args.GetTensors(ArgType::Input);

			//get sum axis
			vector<int> axes;
			for (int i = 0; i < node->data.size(); i++) {
				axes.push_back((int)node->data[i]);
			}

			Tensor* result;
			if (node->name == "dim_sum") {
				result = ComputeSum(inputs[0], axes[0]);
			} else if (node->name == "dim_norm") {
				result = ComputeNorm(inputs[0], axes[0]);
			} else if (node->name == "dim_max") {
				result = ComputeMax(inputs[0], axes[0]);
			} else if (node->name == "dim_min") {
				result = ComputeMin(inputs[0], axes[0]);
			} else if (node->name == "dim_mean") {
				result = ComputeMean(inputs[0], axes[0]);
			} else if (node->name == "dim_product") {
				result = ComputeProduct(inputs[0], axes[0]);
			} else if (node->name == "dim_any") {
				result = ComputeAny(inputs[0], axes[0]);
			} else if (node->name == "dim_all") {
				result = ComputeAll(inputs[0], axes[0]);
			} else if (node->name == "dim_prefix_sum") {
				result = ComputePrefixSum(inputs[0], axes[0]);
			} else if (node->name == "transpose") {
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
				result = Transpose(inputs[0], permutation);
			} else if (node->name == "dot") {
				result = ComputeDot(inputs[0], inputs[1], axes[0]);
			} else if (node->name == "matmul") {
				result = ComputeMatMul(inputs[0], inputs[1]);
			} else if (node->name == "unsqueeze") {
				map<int, int> permutation;
				int dim = (int)inputs[0]->GetDimension()+1;
				dim = std::max(dim, axes[0] + 1);
				for(int i = 0; i < dim; i++) {
					if(i == axes[0]) {
						permutation[i] = 0;
					} else if (i < axes[0]) {
						permutation[i] = i + 1;
					} else {
						permutation[i] = i;
					}
				}
				result = Transpose(inputs[0], permutation);
				result->SetDebugName("unsqueezed");
			} else if (node->name == "squeeze") {
				map<int, int> permutation;
				int dim = (int)inputs[0]->GetDimension() - 1;
				for(int i = 0; i < dim; i++) {
					if(i < axes[0]) {
						permutation[i] = i;
					} else {
						permutation[i] = i + 1;
					}
				}
				result = Transpose(inputs[0], permutation);
				result->SetDebugName("squeezed");
			} else if (node->name == "dim_reverse") {
				result = ReverseDim(inputs[0], axes[0]);
			} else {
				throw std::runtime_error("Unknown algorithmic primitive " + node->name);
			}

			//replace the node with the sum
			node->MakeOutputsUseGivenNode(result->node_);

			ShapeCompareResult shape_result = CompareShape(node, result->node_, true);
			if (!shape_result.compatible) {
				throw std::runtime_error("Algorithmic primitive " + node->name + " at " + node->debug_name + " has incompatible shapes");
			}

			//copy over all memory flags to the new node
			result->node_->flags.copy_all_given(node->flags, {NodeProp::InputMemory, NodeProp::OutputMemory});

			if (node->debug_name != "") {
				result->node_->debug_name = node->debug_name;
			}
		});

		//mark the node for removal
		nodes_to_remove.insert(node);

		UpdateGraph();
	}

	// remove all nodes that are not used
	for (auto* node : nodes_to_remove) {
		RemoveNode(node);
	}

	UpdateGraph();
}

} // namespace TensorFrost
