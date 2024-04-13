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

	m.def("float", [](const PyTensor& t) { return PT(Tensor::tofloat(T(t))); });
	m.def("uint", [](const PyTensor& t) { return PT(Tensor::touint(T(t))); });
	m.def("int", [](const PyTensor& t) { return PT(Tensor::toint(T(t))); });
	m.def("bool", [](const PyTensor& t) { return PT(Tensor::tobool(T(t))); });

	BINARY_FUNCTION(min);
	BINARY_FUNCTION(max);
	BINARY_FUNCTION(pow);
	BINARY_FUNCTION(atan2);
	BINARY_FUNCTION(modf);

	TERNARY_FUNCTION(clamp);
	TERNARY_FUNCTION(fma);
	TERNARY_FUNCTION(lerp);
	TERNARY_FUNCTION(select);
	TERNARY_FUNCTION(smoothstep);

	m.def("scatterAdd", [](const TensorView& t, const PyTensor& t2) {
		Tensor::ScatterAdd(*t.value, T(t2), t.indices);
	});

	m.def("scatterAddPrev", [](const TensorView& t, const PyTensor& t2) {
		return PT(Tensor::ScatterAddPrev(*t.value, T(t2), t.indices));
	});

	m.def("scatterMin", [](const TensorView& t, const PyTensor& t2) {
		Tensor::ScatterMin(*t.value, T(t2), t.indices);
	});

	m.def("scatterMax", [](const TensorView& t, const PyTensor& t2) {
		Tensor::ScatterMax(*t.value, T(t2), t.indices);
	});

	m.def("scatterOr", [](const TensorView& t, const PyTensor& t2) {
		Tensor::ScatterOr(*t.value, T(t2), t.indices);
	});

	m.def("scatterAnd", [](const TensorView& t, const PyTensor& t2) {
		Tensor::ScatterAnd(*t.value, T(t2), t.indices);
	});

	m.def("scatterXor", [](const TensorView& t, const PyTensor& t2) {
		Tensor::ScatterXor(*t.value, T(t2), t.indices);
	});

	m.def("buffer", [](py::list shape, DataType type) {
		    return PT(Tensor::Memory(TensorsFromList(shape), type));
	}, py::arg("shape"), py::arg("type") = DataType::Float);
	m.def("buffer", [](std::vector<int> shape, DataType type) {
		    return PT(Tensor::Memory(shape, type));
	}, py::arg("shape"), py::arg("type") = DataType::Float);

	m.def("zeros", [](py::list shape, DataType type) {
		switch (type)
		{
			case DataType::Float:
				return PT(Tensor::Constant(TensorsFromList(shape), 0.0f));
			case DataType::Uint:
				return PT(Tensor::Constant(TensorsFromList(shape), 0u));
			case DataType::Int:
				return PT(Tensor::Constant(TensorsFromList(shape), 0));
			default:
				return PT(Tensor::Constant(TensorsFromList(shape), 0.0f));
		}
	}, py::arg("shape"), py::arg("type") = DataType::Float);
	m.def("zeros", [](std::vector<int> shape, DataType type) {
		switch (type)
		{
			case DataType::Float:
				return PT(Tensor::Constant(shape, 0.0f));
			case DataType::Uint:
				return PT(Tensor::Constant(shape, 0u));
			case DataType::Int:
				return PT(Tensor::Constant(shape, 0));
			default:
				return PT(Tensor::Constant(shape, 0.0f));
		}
	}, py::arg("shape"), py::arg("type") = DataType::Float);

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

	m.def("input", [](std::vector<int> shape, DataType type) {
		return PT(Tensor::Input(shape, type));
	}, py::arg("shape"), py::arg("type") = DataType::Float);
	m.def("input", [](py::list shape, DataType type) {
		return PT(Tensor::Input(TensorsFromList(shape), type));
	}, py::arg("shape"), py::arg("type") = DataType::Float);	

	m.def("index", [](int dim, py::list shape) {
		return PT(Tensor::Index(TensorsFromList(shape), dim));
	});

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

	m.def(
	    "sum",
	    [](const PyTensor& t, const int axis) { return PT(Tensor::Sum(T(t), axis)); },
	    py::arg("t"), py::arg("axis") = -1,  "Sum the elements of the tensor along the axis");

	m.def(
	    "loop",
	    [](const py::function& body, const PyTensor& begin, const PyTensor& end,
	       const PyTensor& step) {
		    // wrap the function to convert the PyTensor to Tensor
		    std::function<void(const Tensor&)> f2 = [&body](const Tensor& t) {
			    py::gil_scoped_acquire acquire;
			    body(PT(t));
		    };

		    Tensor::Loop(T(begin), T(end), T(step), f2);
	    },
	    py::arg("begin") = 0, py::arg("end"), py::arg("step") = 1,
	    py::arg("body"));

	m.def("if_cond", [](const PyTensor& condition, const py::function& true_body) {
		std::function<void()> f = [&true_body]() {
			py::gil_scoped_acquire acquire;
			true_body();
		};
		Tensor::If(T(condition), f);
	}, py::arg("condition"), py::arg("true_body"));

	m.def("if_cond", [](const PyTensor& condition, const py::function& true_body, const py::function& false_body) {
		std::function<void()> f1 = [&true_body]() {
			py::gil_scoped_acquire acquire;
			true_body();
		};
		std::function<void()> f2 = [&false_body]() {
			py::gil_scoped_acquire acquire;
			false_body();
		};
		Tensor::If(T(condition), f1, f2);
	}, py::arg("condition"), py::arg("true_body"), py::arg("false_body"));

	m.def("break_loop", []() { Tensor::Break(); });
	m.def("continue_loop", []() { Tensor::Continue(); });


	m.def("kernel", [](py::list shape, const py::function& body) {
		// wrap the function to convert the PyTensor to Tensor
		std::function<void(const vector<Tensor*>&)> f2 = [&body](const vector<Tensor*>& tensors) {
			py::gil_scoped_acquire acquire;
			PyTensors py_tensors = PyTensorsFromVector(tensors);
			body(py_tensors);
		};

		Tensors shape_tensors = TensorsFromList(shape);

		Tensor::Kernel(shape_tensors, f2);
	}, py::arg("shape"), py::arg("body"));
}

}  // namespace TensorFrost
