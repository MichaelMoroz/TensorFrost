#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

#define DEFINE_OPERATOR(opname, op)                                            \
	py_tensor.def("__" #opname "__", [](const PyTensor& t, const PyTensor& t2) { \
		return PT(T(t) op T(t2));                                                  \
	});                                                                          \
	py_tensor.def("__r" #opname "__", [](const PyTensor& t, const float f) {     \
		return PT(Tensor::Constant(f) op T(t));                                    \
	});

void PyTensorDefinition(py::module& /*m*/, py::class_<PyTensor>& py_tensor) {
	// initializers
	py_tensor.def(py::init<const TensorView&>());
	py_tensor.def(py::init<float>());
	py_tensor.def(py::init<int>());
	py_tensor.def(py::init<unsigned int>());

	// properties
	py_tensor.def_property_readonly(
	    "shape", [](const PyTensor& t) { return t.Get().GetShape(); });

	py_tensor.def_property_readonly(
	    "type", [](const PyTensor& t) { return t.Get().type; });

	py_tensor.def("index",
	              [](const PyTensor& t, int dim) { return PT(T(t).Index(dim)); });

	// getter
	py_tensor.def("__getitem__", [](const PyTensor& t, py::tuple indices_tuple) {
		Tensors indices = TensorsFromTuple(indices_tuple);
		return TensorView(&t.Get(), std::move(indices));
	});

	// setter
	py_tensor.def("__setitem__", [](const PyTensor& t, py::tuple indices_tuple,
	                                const PyTensor& t2) {
		Tensors indices = TensorsFromTuple(indices_tuple);
		Tensor::Store(t.Get(), T(t2), indices);
	});

	// operator overloads
	DEFINE_OPERATOR(add, +);
	DEFINE_OPERATOR(sub, -);
	DEFINE_OPERATOR(mul, *);
	DEFINE_OPERATOR(div, /);
	DEFINE_OPERATOR(mod, %);
	// negative
	py_tensor.def("__neg__", [](const PyTensor& t) { return PT(-T(t)); });
	// comparison
	DEFINE_OPERATOR(eq, ==);
	DEFINE_OPERATOR(ne, !=);
	DEFINE_OPERATOR(lt, <);
	DEFINE_OPERATOR(le, <=);
	DEFINE_OPERATOR(gt, >);
	DEFINE_OPERATOR(ge, >=);
	// logical
	DEFINE_OPERATOR(and, &&);
	DEFINE_OPERATOR(or, ||);
	py_tensor.def("__not__", [](const PyTensor& t) { return PT(!T(t)); });
	// bitwise
	DEFINE_OPERATOR(xor, ^);
	DEFINE_OPERATOR(lshift, <<);
	DEFINE_OPERATOR(rshift, >>);
	DEFINE_OPERATOR(and_, &);
	DEFINE_OPERATOR(or_, |);
	py_tensor.def("__invert__", [](const PyTensor& t) { return PT(~T(t)); });
	// power operator
	py_tensor.def("__pow__", [](const PyTensor& t, const PyTensor& t2) {
		return PT(Tensor::pow(T(t), T(t2)));
	});
	py_tensor.def("__pow__", [](const PyTensor& t, float f) {
		return PT(Tensor::pow(T(t), Tensor::Constant(f)));
	});
	py_tensor.def("__rpow__", [](const PyTensor& t, float f) {
		return PT(Tensor::pow(Tensor::Constant(f), T(t)));
	});
	// end power operator
	// end operator overloads
	;
}

}  // namespace TensorFrost