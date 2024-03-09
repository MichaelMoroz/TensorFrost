#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

#define DEFINE_OPERATOR(opname, op)                                            \
	tensor_view.def("__" #opname "__",                                           \
	                [](const TensorView& t, const TensorView& t2) {              \
		                return PT(T(PyTensor(t)) op T(PyTensor(t2)));              \
	                });                                                          \
	tensor_view.def("__" #opname "__",                                           \
	                [](const TensorView& t, const PyTensor& t2) {                \
		                return PT(T(PyTensor(t)) op T(t2));                        \
	                });                                                          \
	tensor_view.def("__" #opname "__", [](const TensorView& t, const float f) {  \
		return PT(T(PyTensor(t)) op Tensor::Constant(f));                          \
	});                                                                          \
	tensor_view.def("__r" #opname "__",                                          \
	                [](const TensorView& t, const TensorView& t2) {              \
		                return PT(T(PyTensor(t2)) op T(PyTensor(t)));              \
	                });                                                          \
	tensor_view.def("__r" #opname "__",                                          \
	                [](const TensorView& t, const PyTensor& t2) {                \
		                return PT(T(t2) op T(PyTensor(t)));                        \
	                });                                                          \
	tensor_view.def("__r" #opname "__", [](const TensorView& t, const float f) { \
		return PT(Tensor::Constant(f) op T(PyTensor(t)));                          \
	});

void TensorViewDefinition(py::module& /*m*/,
                          py::class_<TensorView>& tensor_view) {
	DEFINE_OPERATOR(add, +);
	DEFINE_OPERATOR(sub, -);
	DEFINE_OPERATOR(mul, *);
	DEFINE_OPERATOR(div, /);
	DEFINE_OPERATOR(truediv, /);
	DEFINE_OPERATOR(mod, %);
	DEFINE_OPERATOR(eq, ==);
	DEFINE_OPERATOR(ne, !=);
	DEFINE_OPERATOR(lt, <);
	DEFINE_OPERATOR(le, <=);
	DEFINE_OPERATOR(gt, >);
	DEFINE_OPERATOR(ge, >=);
	DEFINE_OPERATOR(and, &&);
	DEFINE_OPERATOR(or, ||);
	DEFINE_OPERATOR(xor, ^);
	DEFINE_OPERATOR(lshift, <<);
	DEFINE_OPERATOR(rshift, >>);
	DEFINE_OPERATOR(and_, &);
	DEFINE_OPERATOR(or_, |);                                                    

	tensor_view.def("__neg__",
	                [](const TensorView& t) { return PT(-T(PyTensor(t))); });
	tensor_view.def("__not__",
	                [](const TensorView& t) { return PT(!T(PyTensor(t))); });
	tensor_view.def("__invert__",
	                [](const TensorView& t) { return PT(~T(PyTensor(t))); });
	tensor_view.def("__pow__", [](const TensorView& t, const TensorView& t2) {
		return PT(Tensor::pow(PyTensor(t).Get(), PyTensor(t2).Get()));
	});
	tensor_view.def("__pow__", [](const TensorView& t, const float f) {
		return PT(Tensor::pow(PyTensor(t).Get(), Tensor::Constant(f)));
	});
	tensor_view.def("__iadd__", [](const TensorView& t, const PyTensor& t2) {
		Tensor::ScatterAdd(*t.value, t2.Get(), t.indices);
	});
	tensor_view.def("__setitem__", [](const TensorView& t, const PyTensor& t2) {
		Tensor::Store(*t.value, t2.Get(), t.indices);
	});
}

}  // namespace TensorFrost