#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

#define UNARY_FUNCTION(name) \
	m.def(#name, [](const PyTensor& t) { return PT(Tensor::name(T(t))); })

#define BINARY_FUNCTION(name)                              \
	m.def(#name, [](const PyTensor& t, const PyTensor& t2) { \
		return PT(Tensor::name(T(t), T(t2)));                  \
	})

#define TERNARY_FUNCTION(name)                                                 \
	m.def(#name, [](const PyTensor& t, const PyTensor& t2, const PyTensor& t3) { \
		return PT(Tensor::name(T(t), T(t2), T(t3)));                               \
	})

void TensorFunctionsDefinition(py::module& m) {
	UNARY_FUNCTION(copy);
	UNARY_FUNCTION(abs);
	UNARY_FUNCTION(ceil);
	UNARY_FUNCTION(floor);
	UNARY_FUNCTION(round);
	UNARY_FUNCTION(trunc);
	UNARY_FUNCTION(sign);
	UNARY_FUNCTION(frac);
	UNARY_FUNCTION(sin);
	UNARY_FUNCTION(cos);
	UNARY_FUNCTION(tan);
	UNARY_FUNCTION(asin);
	UNARY_FUNCTION(acos);
	UNARY_FUNCTION(atan);
	UNARY_FUNCTION(sinh);
	UNARY_FUNCTION(cosh);
	UNARY_FUNCTION(tanh);
	UNARY_FUNCTION(exp);
	UNARY_FUNCTION(exp2);
	UNARY_FUNCTION(log);
	UNARY_FUNCTION(log2);
	UNARY_FUNCTION(sqrt);
	UNARY_FUNCTION(sqr);
	UNARY_FUNCTION(rsqrt);
	UNARY_FUNCTION(rcp);

	UNARY_FUNCTION(pcg);
	UNARY_FUNCTION(pcgf);
	UNARY_FUNCTION(reversebits);

	m.def("float", [](const PyTensor& t) { return PT(Tensor::tofloat(T(t))); });
	m.def("uint", [](const PyTensor& t) { return PT(Tensor::touint(T(t))); });
	m.def("int", [](const PyTensor& t) { return PT(Tensor::toint(T(t))); });
	m.def("bool", [](const PyTensor& t) { return PT(Tensor::tobool(T(t))); });

	BINARY_FUNCTION(min);
	BINARY_FUNCTION(max);
	BINARY_FUNCTION(pow);
	BINARY_FUNCTION(atan2);
	BINARY_FUNCTION(modf);

	BINARY_FUNCTION(grad);

	TERNARY_FUNCTION(clamp);
	TERNARY_FUNCTION(fma);
	TERNARY_FUNCTION(lerp);
	TERNARY_FUNCTION(select);
	TERNARY_FUNCTION(smoothstep);

	m.def("scatterAdd", [](const PyTensor& t, const PyTensor& t2) {
		Tensor::ScatterAdd(*t.Value(), T(t2), t.Indices());
	});

	m.def("scatterAddPrev", [](const PyTensor& t, const PyTensor& t2) {
		return PT(Tensor::ScatterAddPrev(*t.Value(), T(t2), t.Indices()));
	});

	m.def("scatterMin", [](const PyTensor& t, const PyTensor& t2) {
		Tensor::ScatterMin(*t.Value(), T(t2), t.Indices());
	});

	m.def("scatterMax", [](const PyTensor& t, const PyTensor& t2) {
		Tensor::ScatterMax(*t.Value(), T(t2), t.Indices());
	});

	m.def("scatterOr", [](const PyTensor& t, const PyTensor& t2) {
		Tensor::ScatterOr(*t.Value(), T(t2), t.Indices());
	});

	m.def("scatterAnd", [](const PyTensor& t, const PyTensor& t2) {
		Tensor::ScatterAnd(*t.Value(), T(t2), t.Indices());
	});

	m.def("scatterXor", [](const PyTensor& t, const PyTensor& t2) {
		Tensor::ScatterXor(*t.Value(), T(t2), t.Indices());
	});

	m.def("buffer", [](py::list shape, TFType type) {
		    return PT(Tensor::Memory(TensorsFromList(shape), type));
	}, py::arg("shape"), py::arg("type") = TFType::Float);
	m.def("buffer", [](std::vector<int> shape, TFType type) {
		    return PT(Tensor::Memory(shape, type));
	}, py::arg("shape"), py::arg("type") = TFType::Float);

	m.def("zeros", [](py::list shape, TFType type) {
		switch (type)
		{
			case TFType::Float:
				return PT(Tensor::Constant(TensorsFromList(shape), 0.0f));
			case TFType::Uint:
				return PT(Tensor::Constant(TensorsFromList(shape), 0u));
			case TFType::Int:
				return PT(Tensor::Constant(TensorsFromList(shape), 0));
			default:
				return PT(Tensor::Constant(TensorsFromList(shape), 0.0f));
		}
	}, py::arg("shape"), py::arg("type") = TFType::Float);
	m.def("zeros", [](std::vector<int> shape, TFType type) {
		switch (type)
		{
			case TFType::Float:
				return PT(Tensor::Constant(shape, 0.0f));
			case TFType::Uint:
				return PT(Tensor::Constant(shape, 0u));
			case TFType::Int:
				return PT(Tensor::Constant(shape, 0));
			default:
				return PT(Tensor::Constant(shape, 0.0f));
		}
	}, py::arg("shape"), py::arg("type") = TFType::Float);

	m.def("const", [](float value, py::list shape) {
		return PT(Tensor::Constant(TensorsFromList(shape), value));
	});
	m.def("const", [](float value, std::vector<int> shape) {
		return PT(Tensor::Constant(shape, value));
	}, py::arg("value"), py::arg("shape") = std::vector<int>{});

	m.def("const", [](int value, py::list shape) {
		return PT(Tensor::Constant(TensorsFromList(shape), value));
	});
	m.def("const", [](int value, std::vector<int> shape) {
		return PT(Tensor::Constant(shape, value));
	}, py::arg("value"), py::arg("shape") = std::vector<int>{});

	m.def("input", [](std::vector<int> shape, TFType type) {
		return PT(Tensor::Input(shape, type));
	}, py::arg("shape"), py::arg("type") = TFType::Float);
	m.def("input", [](py::list shape, TFType type) {
		return PT(Tensor::Input(TensorsFromList(shape), type));
	}, py::arg("shape"), py::arg("type") = TFType::Float);

	m.def("index", [](int dim, py::list shape) {
		return PT(Tensor::Index(TensorsFromList(shape), dim));
	});

	m.def("get_copy", [](const PyTensor& t) { return PT(*Tensor::GetCopy(T(t))); });

	m.def("indices", [](py::list shape) {
		Tensors shape_tensors = TensorsFromList(shape);
		py::tuple indices = py::tuple(shape_tensors.size());
		for (int i = 0; i < shape_tensors.size(); i++) {
			auto t = PT(Tensor::Index(shape_tensors, i));
			indices[i] = t;
		}
		return indices;
	});

	m.def("indices", [](std::vector<int> shape) {
		py::tuple indices = py::tuple(shape.size());
		for (int i = 0; i < shape.size(); i++) {
			auto t = PT(Tensor::Index(shape, i));
			indices[i] = t;
		}
		return indices;
	});

	m.def("index_grid", [](py::list begin, py::list end) {
		Tensors begin_tensors = TensorsFromList(begin);
		Tensors end_tensors = TensorsFromList(end);
		Tensors index_grid = Tensor::IndexGrid(begin_tensors, end_tensors);

		py::tuple indices = py::tuple(begin.size());
		for (int i = 0; i < index_grid.size(); i++) {
			indices[i] = PT(*index_grid[i]);
		}
		return indices;
	});

	m.def("index_grid", [](py::list begin, py::list end, py::list step) {
		Tensors begin_tensors = TensorsFromList(begin);
		Tensors end_tensors = TensorsFromList(end);
		Tensors step_tensors = TensorsFromList(step);
		Tensors index_grid = Tensor::IndexGrid(begin_tensors, end_tensors, step_tensors);

		py::tuple indices = py::tuple(begin.size());
		for (int i = 0; i < index_grid.size(); i++) {
			indices[i] = PT(*index_grid[i]);
		}
		return indices;
	});

	m.def("reshape", [](const PyTensor& t, py::list shape) {
		return PT(Tensor::Reshape(T(t), TensorsFromList(shape)));
	});

	//algorithm functions
	m.def("sum", [](const PyTensor& t, const int axis) { return PT(Tensor::Sum(T(t), axis)); },
	    py::arg("t"), py::arg("axis") = -1,  "Sum the elements of the tensor along the axis");

	m.def("norm", [](const PyTensor& t, const int axis) { return PT(Tensor::Norm(T(t), axis)); },
	    py::arg("t"), py::arg("axis") = -1, "Compute the norm of the tensor along the axis");

	m.def("mean", [](const PyTensor& t, const int axis) { return PT(Tensor::Mean(T(t), axis)); },
	    py::arg("t"), py::arg("axis") = -1, "Compute the mean of the tensor along the axis");

	m.def("min", [](const PyTensor& t, const int axis) { return PT(Tensor::Min(T(t), axis)); },
	    py::arg("t"), py::arg("axis") = -1, "Compute the min of the tensor along the axis");

	m.def("max", [](const PyTensor& t, const int axis) { return PT(Tensor::Max(T(t), axis)); },
	    py::arg("t"), py::arg("axis") = -1, "Compute the max of the tensor along the axis");

	m.def("prefix_sum", [](const PyTensor& t, const int axis) { return PT(Tensor::PrefixSum(T(t), axis)); },
	    py::arg("t"), py::arg("axis") = -1, "Compute the prefix sum of the tensor along the axis");

	m.def("reverse", [](const PyTensor& t, const int axis) { return PT(Tensor::Reverse(T(t), axis)); },
	    py::arg("t"), py::arg("axis") = -1, "Reverse the tensor along the axis");

	m.def("transpose", [](const PyTensor& t, int dim1, int dim2) {
		return PT(Tensor::Transpose(T(t), dim1, dim2));
	}, py::arg("t"), py::arg("dim1") = -2, py::arg("dim2") = -1, "Transpose the tensor");

	m.def("unsqueeze", [](const PyTensor& t, int dim) {
		return PT(Tensor::Unsqueeze(T(t), dim));
	}, py::arg("t"), py::arg("dim") = -1, "Unsqueeze the tensor");

	m.def("dot", [](const PyTensor& t, const PyTensor& t2, int axis) {
		return PT(Tensor::Dot(T(t), T(t2), axis));
	}, py::arg("t"), py::arg("t2"), py::arg("axis") = -1, "Dot product of two tensors");

	m.def("matmul", [](const PyTensor& t, const PyTensor& t2) {
		return PT(Tensor::Matmul(T(t), T(t2)));
	}, py::arg("t"), py::arg("t2"), "Matrix multiplication of two tensors");
}

}  // namespace TensorFrost
