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

	m.def("asfloat", [](const PyTensor& t) { return PT(Tensor::asfloat(T(t))); });
	m.def("asuint", [](const PyTensor& t) { return PT(Tensor::asuint(T(t))); });
	m.def("asint", [](const PyTensor& t) { return PT(Tensor::asint(T(t))); });

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
		    return PT(Tensor::Memory(Reverse(TensorsFromList(shape)), type));
	}, py::arg("shape"), py::arg("type") = TFType::Float);
	m.def("buffer", [](std::vector<int> shape, TFType type) {
		    return PT(Tensor::Memory(Reverse(shape), type));
	}, py::arg("shape"), py::arg("type") = TFType::Float);

	m.def("local_buffer", [](int size, TFType type) {
		return PT(Tensor::LocalMemory(size, type));
	}, py::arg("size"), py::arg("type") = TFType::Float);
	m.def("group_buffer", [](int size, TFType type) {
		return PT(Tensor::GroupMemory(size, type));
	}, py::arg("size"), py::arg("type") = TFType::Float);
	m.def("group_barrier", []() {
		Tensor::GroupBarrier();
	});

	m.def("zeros", [](py::list shape, TFType type) {
		return PT(Tensor::Constant(0u, Reverse(TensorsFromList(shape)), type));
	}, py::arg("shape"), py::arg("type") = TFType::Float);

	m.def("const", [](float value, py::list shape) {
		return PT(Tensor::Constant(Reverse(TensorsFromList(shape)), value));
	});
	m.def("const", [](float value, std::vector<int> shape) {
		return PT(Tensor::Constant(Reverse(shape), value));
	}, py::arg("value"), py::arg("shape") = std::vector<int>{});

	m.def("const", [](int value, py::list shape) {
		return PT(Tensor::Constant(Reverse(TensorsFromList(shape)), value));
	});
	m.def("const", [](int value, std::vector<int> shape) {
		return PT(Tensor::Constant(Reverse(shape), value));
	}, py::arg("value"), py::arg("shape") = std::vector<int>{});

	m.def("input", [](std::vector<int> shape, TFType type) {
		return PT(Tensor::Input(Reverse(shape), type));
	}, py::arg("shape"), py::arg("type") = TFType::Float);
	m.def("input", [](py::list shape, TFType type) {
		return PT(Tensor::Input(Reverse(TensorsFromList(shape)), type));
	}, py::arg("shape"), py::arg("type") = TFType::Float);

	m.def("index", [](int dim, py::list shape) {
		return PT(Tensor::Index(Reverse(TensorsFromList(shape)), dim));
	});

	m.def("get_copy", [](const PyTensor& t) { return PT(*Tensor::GetCopy(T(t))); });

	m.def("indices", [](py::list shape) {
		Tensors shape_tensors = Reverse(TensorsFromList(shape));
		int dim = (int)shape_tensors.size();
		py::tuple indices = py::tuple(shape_tensors.size());
		for (int i = 0; i < shape_tensors.size(); i++) {
			auto t = PT(Tensor::Index(shape_tensors, dim - i - 1));
			indices[i] = t;
		}
		return indices;
	});

	m.def("indices", [](std::vector<int> shape) {
		py::tuple indices = py::tuple(shape.size());
		int dim = (int)shape.size();
		for (int i = 0; i < shape.size(); i++) {
			auto t = PT(Tensor::Index(Reverse(shape), dim - i - 1));
			indices[i] = t;
		}
		return indices;
	});

	m.def("index_grid", [](py::list begin, py::list end) {
		Tensors begin_tensors = Reverse(TensorsFromList(begin));
		Tensors end_tensors = Reverse(TensorsFromList(end));
		Tensors index_grid = Reverse(Tensor::IndexGrid(begin_tensors, end_tensors));

		py::tuple indices = py::tuple(begin.size());
		for (int i = 0; i < index_grid.size(); i++) {
			indices[i] = PT(*index_grid[i]);
		}
		return indices;
	});

	m.def("index_grid", [](py::list begin, py::list end, py::list step) {
		Tensors begin_tensors = Reverse(TensorsFromList(begin));
		Tensors end_tensors = Reverse(TensorsFromList(end));
		Tensors step_tensors = Reverse(TensorsFromList(step));
		Tensors index_grid = Reverse(Tensor::IndexGrid(begin_tensors, end_tensors, step_tensors));

		py::tuple indices = py::tuple(begin.size());
		for (int i = 0; i < index_grid.size(); i++) {
			indices[i] = PT(*index_grid[i]);
		}
		return indices;
	});

	m.def("reshape", [](const PyTensor& t, py::list shape) {
		return PT(Tensor::Reshape(T(t), Reverse(TensorsFromList(shape))));
	});
	m.def("assert_tensor", [](const PyTensor& t, py::list target_shape, TFType target_type) {
		return PT(Tensor::Assert(T(t), Reverse(TensorsFromList(target_shape)), target_type));
	});
	m.def("split_dim", [](const PyTensor& t, const int split_size, const int axis) {
		return PT(Tensor::SplitDim(T(t), split_size, -axis-1));
	}, py::arg("t"), py::arg("split_size"), py::arg("axis") = -1);
	m.def("merge_dim", [](const PyTensor& t, const int axis, const PyTensor* target_size) {
		const Tensor* target_size_ptr = target_size ? &T(*target_size) : nullptr;
		return PT(Tensor::MergeDim(T(t), -axis-1, target_size_ptr));
	}, py::arg("t"), py::arg("axis") = -1, py::arg("target_size") = nullptr);
	m.def("repeat", [](const PyTensor& t, const PyTensor& repeats, const int axis) {
		return PT(Tensor::Repeat(T(t), T(repeats), -axis-1));
	}, py::arg("t"), py::arg("repeats"), py::arg("axis") = -1);

	//algorithm functions
	m.def("sum", [](const PyTensor& t, const int axis) { return PT(Tensor::Sum(T(t), -axis-1)); },
	    py::arg("t"), py::kw_only(), py::arg("axis") = -1,  "Sum the elements of the tensor along the axis");

	m.def("norm", [](const PyTensor& t, const int axis) { return PT(Tensor::Norm(T(t), -axis-1)); },
	    py::arg("t"), py::kw_only(), py::arg("axis") = -1, "Compute the norm of the tensor along the axis");

	m.def("mean", [](const PyTensor& t, const int axis) { return PT(Tensor::Mean(T(t), -axis-1)); },
	    py::arg("t"), py::kw_only(), py::arg("axis") = -1, "Compute the mean of the tensor along the axis");

	m.def("min", [](const PyTensor& t, const int axis) { return PT(Tensor::Min(T(t), -axis-1)); },
	    py::arg("t"), py::kw_only(), py::arg("axis") = -1, "Compute the min of the tensor along the axis");

	m.def("max", [](const PyTensor& t, const int axis) { return PT(Tensor::Max(T(t), -axis-1)); },
	    py::arg("t"), py::kw_only(), py::arg("axis") = -1, "Compute the max of the tensor along the axis");

	m.def("any", [](const PyTensor& t, const int axis) { return PT(Tensor::Any(T(t), -axis-1)); },
	    py::arg("t"), py::kw_only(), py::arg("axis") = -1, "Do an OR operation along the axis");

	m.def("all", [](const PyTensor& t, const int axis) { return PT(Tensor::All(T(t), -axis-1)); },
	    py::arg("t"), py::kw_only(), py::arg("axis") = -1, "Do an AND operation along the axis");

	m.def("prefix_sum", [](const PyTensor& t, const int axis) { return PT(Tensor::PrefixSum(T(t), -axis-1)); },
	    py::arg("t"), py::kw_only(), py::arg("axis") = -1, "Compute the prefix sum of the tensor along the axis");

	m.def("reverse", [](const PyTensor& t, const int axis) { return PT(Tensor::Reverse(T(t), -axis-1)); },
	    py::arg("t"), py::kw_only(), py::arg("axis") = -1, "Reverse the tensor along the axis");

	m.def("transpose", [](const PyTensor& t, int dim1, int dim2) {
		return PT(Tensor::Transpose(T(t), -dim1-1, -dim2-1));
	}, py::arg("t"), py::kw_only(), py::arg("dim1") = -2, py::arg("dim2") = -1, "Transpose the tensor");

	m.def("unsqueeze", [](const PyTensor& t, int dim) {
		return PT(Tensor::Unsqueeze(T(t), -dim-1));
	}, py::arg("t"), py::kw_only(), py::arg("axis") = -1, "Unsqueeze the tensor");

	m.def("squeeze", [](const PyTensor& t, int dim) {
		return PT(Tensor::Squeeze(T(t), -dim-1));
	}, py::arg("t"), py::kw_only(), py::arg("axis") = -1, "Squeeze the tensor");

	m.def("dot", [](const PyTensor& t, const PyTensor& t2, int axis) {
		return PT(Tensor::Dot(T(t), T(t2), -axis-1));
	}, py::arg("t"),  py::arg("t2"), py::kw_only(), py::arg("axis") = -1, "Dot product of two tensors");

	m.def("matmul", [](const PyTensor& t, const PyTensor& t2) {
		return PT(Tensor::Matmul(T(t), T(t2)));
	}, py::arg("t"), py::arg("t2"), "Matrix multiplication of two tensors");

	m.def("region_begin", [](const std::string& name) {
		Tensor::BeginRegion(name);
	}, py::arg("name"), "Begin a debug region");

	m.def("region_end", [](const std::string& name) {
		Tensor::EndRegion(name);
	}, py::arg("name"), "End a debug region");

	m.def("register_custom_operation", [](const std::string& name, vector<string> overloads, py::function impl, py::function vjp) {
		auto cpp_impl = [impl](Tensors& output, map<int, const Tensor*> inputs, const Tensor* tensor, vector<int> axes) {
			py::list input_list;
			for (auto& [id, tensor] : inputs) {
				input_list.append(PT(*tensor));
			}
			py::list output_list = impl(input_list, PT(*tensor), axes).cast<py::list>();
			for (int i = 0; i < output_list.size(); i++) {
				PyTensor* t = output_list[i].cast<PyTensor*>();
				output.push_back(&t->Get());
			}
		};

		auto cpp_vjp = [vjp](map<int, const Tensor*> inputs, const Tensor* gradient, const Tensor* tensor) {
			py::list input_list;
			for (auto& [id, tensor] : inputs) {
				input_list.append(PT(*tensor));
			}
			py::list output_list = vjp(input_list, PT(*gradient), PT(*tensor)).cast<py::list>();
			Tensors gradients;
			for (int i = 0; i < output_list.size(); i++) {
				PyTensor* t = output_list[i].cast<PyTensor*>();
				gradients.push_back(&t->Get());
			}
			return gradients;
		};

		RegisterAlgorithmicPrimitive(name, overloads, cpp_impl, cpp_vjp);
	}, py::arg("name"), py::arg("overloads"), py::arg("impl"), py::arg("vjp"), "Register a custom operation");

	m.def("custom", [](const std::string& name, py::list inputs, py::list shape) {
		Tensors input_tensors = TensorsFromList(inputs);
		Tensors shape_tensors = Reverse(TensorsFromList(shape));
		return PT(Tensor::CustomOperation(name, input_tensors, shape_tensors));
	}, py::arg("name"), py::arg("inputs"), py::arg("shape"), "Run custom operation");

	m.def("custom", [](const std::string& name, py::list inputs) {
		Tensors input_tensors = TensorsFromList(inputs);
		Tensors shape_tensors = input_tensors[0]->GetShape();
		return PT(Tensor::CustomOperation(name, input_tensors, shape_tensors));
	}, py::arg("name"), py::arg("inputs"), "Run custom operation");

	m.def("print_value", [](const std::string& name, const PyTensor& t) {
		Tensor::PrintValue(name, T(t));
	}, py::arg("name"), py::arg("t"), "Print the value of the tensor");

	m.def("assert_value", [](const std::string& name, const PyTensor& t) {
		Tensor::AssertValue(name, T(t));
	}, py::arg("name"), py::arg("t"), "Assert the value of the tensor");
}

}  // namespace TensorFrost
