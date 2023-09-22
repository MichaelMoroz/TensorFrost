#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

void TensorViewDefinition(py::module& m, py::class_<TensorView>& tensorView) {
#define DEFINE_OPERATOR(opname, op)                                          \
	tensorView.def("__" #opname "__",                                          \
	               [](const TensorView& t, const TensorView& t2) {             \
		               return PT(T(PyTensor(t)) op T(PyTensor(t2)));             \
	               });                                                         \
	tensorView.def("__" #opname "__", [](const TensorView& t, const float f) { \
		return PT(Tensor::Constant(t.value->shape, f) op T(PyTensor(t)));        \
	});

	DEFINE_OPERATOR(add, +);
	DEFINE_OPERATOR(sub, -);
	DEFINE_OPERATOR(mul, *);
	DEFINE_OPERATOR(div, /);
	DEFINE_OPERATOR(mod, %);
	// negative
	tensorView.def("__neg__",
	               [](const TensorView& t) { return PT(-T(PyTensor(t))); });
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
	tensorView.def("__not__",
	               [](const TensorView& t) { return PT(!T(PyTensor(t))); });
	// bitwise
	DEFINE_OPERATOR(xor, ^);
	DEFINE_OPERATOR(lshift, <<);
	DEFINE_OPERATOR(rshift, >>);
	DEFINE_OPERATOR(and_, &);
	DEFINE_OPERATOR(or_, |);
	tensorView.def("__invert__",
	               [](const TensorView& t) { return PT(~T(PyTensor(t))); });

	//** operator overload
	tensorView.def("__pow__", [](const TensorView& t, const TensorView& t2) {
		return PT(Tensor::pow(PyTensor(t).Get(), PyTensor(t2).Get()));
	});
}

}  // namespace TensorFrost