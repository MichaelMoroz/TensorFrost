#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

std::string r_op(std::string name)
{
	return "__r" + name + "__";
}

std::string l_op(std::string name)
{
	return "__" + name + "__";
}

void DefineOperator(std::string pyname, py::class_<PyTensor>& py_tensor, std::function<Tensor&(const Tensor&, const Tensor&)> op) {
	py_tensor.def(l_op(pyname).c_str(), 
	[op](const PyTensor& t, const PyTensor& t2) {
		return PT(op(T(t), T(t2)));
	});
	py_tensor.def(l_op(pyname).c_str(), 
	[op](const PyTensor& t, const float f) {
		return PT(op(T(t), Tensor::Constant(f)));
	});
	py_tensor.def(l_op(pyname).c_str(), 
	[op](const PyTensor& t, const int i) {
		return PT(op(Tensor::Constant(i), T(t)));
	});
	py_tensor.def(r_op(pyname).c_str(),
	[op](const PyTensor& t, const float f) {
		return PT(op(Tensor::Constant(f), T(t)));
	});
	py_tensor.def(r_op(pyname).c_str(),
	[op](const PyTensor& t, const int i) {
		 return PT(op(Tensor::Constant(i), T(t)));
	});
}

#define LAMBDA_OP(op) [](const Tensor& t1, const Tensor& t2) -> Tensor& { return t1 op t2; }

void DefineOperators(py::class_<PyTensor>& py_tensor)
{
	DefineOperator("add", py_tensor, LAMBDA_OP(+));
	DefineOperator("sub", py_tensor, LAMBDA_OP(-));
	DefineOperator("mul", py_tensor, LAMBDA_OP(*));
	DefineOperator("div", py_tensor, LAMBDA_OP(/));
	DefineOperator("truediv", py_tensor, LAMBDA_OP(/));
	DefineOperator("mod", py_tensor, LAMBDA_OP(%));
	DefineOperator("eq", py_tensor, LAMBDA_OP(==));
	DefineOperator("ne", py_tensor, LAMBDA_OP(!=));
	DefineOperator("lt", py_tensor, LAMBDA_OP(<));
	DefineOperator("le", py_tensor, LAMBDA_OP(<=));
	DefineOperator("gt", py_tensor, LAMBDA_OP(>));
	DefineOperator("ge", py_tensor, LAMBDA_OP(>=));
	DefineOperator("and", py_tensor, LAMBDA_OP(&&));
	DefineOperator("or", py_tensor, LAMBDA_OP(||));
	DefineOperator("xor", py_tensor, LAMBDA_OP(^));
	DefineOperator("lshift", py_tensor, LAMBDA_OP(<<));
	DefineOperator("rshift", py_tensor, LAMBDA_OP(>>));
	DefineOperator("and_", py_tensor, LAMBDA_OP(&));
	DefineOperator("or_", py_tensor, LAMBDA_OP(|));

	py_tensor.def("__neg__", [](const PyTensor& t) { return PT(-T(t)); });
	py_tensor.def("__not__", [](const PyTensor& t) { return PT(!T(t)); });
	py_tensor.def("__invert__", [](const PyTensor& t) { return PT(~T(t)); });
	py_tensor.def("__pow__", [](const PyTensor& t, const PyTensor& t2) {
		return PT(Tensor::pow(T(t), T(t2)));
	});
	py_tensor.def("__pow__", [](const PyTensor& t, float f) {
		return PT(Tensor::pow(T(t), Tensor::Constant(f)));
	});
	py_tensor.def("__rpow__", [](const PyTensor& t, float f) {
		return PT(Tensor::pow(Tensor::Constant(f), T(t)));
	});
}

void PyTensorDefinition(py::module& /*m*/, py::class_<PyTensor>& py_tensor) {
	// initializers
	py_tensor.def(py::init<const TensorView&>());
	py_tensor.def(py::init<float>());
	py_tensor.def(py::init<int>());
	py_tensor.def(py::init<unsigned int>());

	// properties
	py_tensor.def_property_readonly("shape", [](const PyTensor& t) {
		return PyTensorsFromTensors(t.Get().GetShape());
	});
	py_tensor.def_property_readonly(
	    "type", [](const PyTensor& t) { return t.Get().type; });
	py_tensor.def_property_readonly("indices", [](const PyTensor& t) {
		int dim = T(t).GetDimension();
		py::print("dim = ", dim);
		py::tuple indices(dim);
		for (int i = 0; i < dim; i++) {
			indices[i] = PT(T(t).Index(i));
		}
		return indices;
	});
	py_tensor.def("index",
	              [](const PyTensor& t, int dim) { return PT(T(t).Index(dim)); });
	

	// operators
	DefineOperators(py_tensor);

	// indexing
	py_tensor.def("__getitem__", [](const PyTensor& t, py::tuple indices_tuple) {
		Tensors indices = TensorsFromTuple(indices_tuple);
		return TensorView(&t.Get(), indices);
	});
	py_tensor.def("__setitem__", [](const PyTensor& t, py::tuple indices_tuple,
	                                const PyTensor& t2) {
		Tensors indices = TensorsFromTuple(indices_tuple);
		Tensor::Store(t.Get(), T(t2), indices);
	});
}

}  // namespace TensorFrost