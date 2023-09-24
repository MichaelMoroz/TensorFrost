#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

#define DEFINE_OPERATOR(opname, op) \
	pyTensor.def("__" #opname "__", [](const PyTensor& t, const PyTensor& t2) { \
		return PT(T(t) op T(t2)); \
	}); \
	pyTensor.def("__r" #opname "__", [](const PyTensor& t, const float f) { \
		return PT(Tensor::Constant(f) op T(t)); \
	}); \

void PyTensorDefinition(py::module& m, py::class_<PyTensor>& pyTensor) {
	// initializers
	pyTensor.def(py::init<const TensorView&>());
	pyTensor.def(py::init<float>());
	pyTensor.def(py::init<int>());
	pyTensor.def(py::init<unsigned int>());

	// properties
	pyTensor.def_property_readonly(
	    "shape", [](const PyTensor& t) { return t.Get().GetShape(); });

	pyTensor.def_property_readonly(
	    "type", [](const PyTensor& t) { return t.Get().type; });

	pyTensor.def("index",
	         [](const PyTensor& t, int dim) { return PT(T(t).Index(dim)); });

	// getter
	pyTensor.def("__getitem__", [](const PyTensor& t, py::tuple indices_tuple) {
		Tensors indices = TensorsFromTuple(indices_tuple);
		return TensorView(&t.Get(), indices);
	});

	// setter
	pyTensor.def("__setitem__", [](const PyTensor& t, py::tuple indices_tuple,
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
	pyTensor.def("__neg__", [](const PyTensor& t) { return PT(-T(t)); });
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
	pyTensor.def("__not__", [](const PyTensor& t) { return PT(!T(t)); });
	// bitwise
	DEFINE_OPERATOR(xor, ^);
	DEFINE_OPERATOR(lshift, <<);
	DEFINE_OPERATOR(rshift, >>);
	DEFINE_OPERATOR(and_, &);
	DEFINE_OPERATOR(or_, |);
	pyTensor.def("__invert__", [](const PyTensor& t) { return PT(~T(t)); });
	// power operator
	pyTensor.def("__pow__", [](const PyTensor& t, const PyTensor& t2) {
		return PT(Tensor::pow(T(t), T(t2)));
	});
	pyTensor.def("__pow__", [](const PyTensor& t, float f) {
		return PT(Tensor::pow(T(t), Tensor::Constant(f)));
	});
	pyTensor.def("__rpow__", [](const PyTensor& t, float f) {
		return PT(Tensor::pow(Tensor::Constant(f), T(t)));
	});
	// end power operator
	// end operator overloads
	;
}

}  // namespace TensorFrost