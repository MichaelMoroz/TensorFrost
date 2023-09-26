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

	BINARY_FUNCTION(min);
	BINARY_FUNCTION(max);
	BINARY_FUNCTION(pow);
	BINARY_FUNCTION(atan2);

	TERNARY_FUNCTION(clamp);
	TERNARY_FUNCTION(fma);
	TERNARY_FUNCTION(lerp);

	m.def("scatterAdd", [](const TensorView& t, const PyTensor& t2) {
		Tensor::ScatterAdd(*t.value, T(t2), t.indices);
	});

	m.def("scatterMin", [](const TensorView& t, const PyTensor& t2) {
		Tensor::ScatterMin(*t.value, T(t2), t.indices);
	});

	m.def("scatterMax", [](const TensorView& t, const PyTensor& t2) {
		Tensor::ScatterMax(*t.value, T(t2), t.indices);
	});

	m.def("zeros", [](py::list shape) {
		return PT(Tensor::Constant(TensorsFromList(shape), 0));
	});

	m.def("const", [](py::list shape, float value) {
		return PT(Tensor::Constant(TensorsFromList(shape), value));
	});

	m.def("input",
	      [](std::vector<int> shape) { return PT(Tensor::Input(shape)); });

	m.def("index", [](int dim, py::list shape) {
		return PT(Tensor::Index(TensorsFromList(shape), dim));
	});
}

}  // namespace TensorFrost
