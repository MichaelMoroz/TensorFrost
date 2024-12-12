#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

void DefineOperator(
    const std::string& pyname, py::class_<PyTensor>& py_tensor,
    const std::function<Tensor&(const Tensor&, const Tensor&)>& op) {
	py_tensor.def(l_op(pyname).c_str(),
	              [op](const PyTensor& t, const PyTensor& t2) {
		              return PT(op(T(t), T(t2)));
	              });
	py_tensor.def(l_op(pyname).c_str(), [op](const PyTensor& t, const float f) {
		return PT(op(T(t), Tensor::Constant(f)));
	});
	py_tensor.def(l_op(pyname).c_str(), [op](const PyTensor& t, const int i) {
		return PT(op(T(t), Tensor::Constant(i)));
	});
	py_tensor.def(r_op(pyname).c_str(), [op](const PyTensor& t, const float f) {
		return PT(op(Tensor::Constant(f), T(t)));
	});
	py_tensor.def(r_op(pyname).c_str(), [op](const PyTensor& t, const int i) {
		return PT(op(Tensor::Constant(i), T(t)));
	});
}

#define LAMBDA_OP(op) \
	[](const Tensor& t1, const Tensor& t2) -> Tensor& { return t1 op t2; }

void DefineOperators(py::class_<PyTensor>& py_tensor) {
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
	py_tensor.def("__matmul__", [](const PyTensor& t, const PyTensor& t2) {
		return PT(Tensor::Matmul(T(t), T(t2)));
	});
}

void PyTensorDefinition(py::module& /*m*/, py::class_<PyTensor>& py_tensor) {
	// initializers
	py_tensor.def(py::init<float>());
	py_tensor.def(py::init<bool>());
	py_tensor.def(py::init<int>());
	py_tensor.def(py::init<unsigned int>());

	// properties
	py_tensor.def_property_readonly("shape", [](const PyTensor& t) {
		return PyTensorsFromTensors(Reverse(t.Get().GetShape()));
	});
	py_tensor.def_property_readonly(
	    "type", [](const PyTensor& t) { return t.Get().GetType(); });
	py_tensor.def_property_readonly("indices", [](const PyTensor& t) {
		int dim = T(t).GetDimension();
		py::tuple indices(dim);
		for (int i = 0; i < dim; i++) {
			indices[i] = PT(T(t).Index(dim - i - 1));
		}
		return indices;
	});
	py_tensor.def("index",[](const PyTensor& t, int dim) {
		  int dims = T(t).GetDimension();
          return PT(T(t).Index(dims - dim - 1));
	});

	py_tensor.def("block_index", [](const PyTensor& t) {
		return PT(T(t).BlockIndex());
	});

	py_tensor.def("block_thread_index", [](const PyTensor& t, int block_dim) {
		return PT(T(t).BlockThreadIndex(block_dim));
	});

	py_tensor.def("detach_grad", [](const PyTensor& t) {
		t.Get().DetachGrad();
		return t;
	});

	py_tensor.def("pass_grad", [](const PyTensor& t) {
		t.Get().PassGrad();
		return t;
	});

	py_tensor.def("stop_fusion", [](const PyTensor& t) {
		t.Get().StopFusion();
		return t;
	});

	py_tensor.def("hint_range", [](const PyTensor& t, py::object min, py::object max) {
		if(t.Get().node_->type == TFType::Float) {
			t.Get().HintRange(py::cast<float>(min), py::cast<float>(max));
		} else {
			t.Get().HintRange(py::cast<int>(min), py::cast<int>(max));
		}
	}, py::arg("min"), py::arg("max"));

	// operators
	DefineOperators(py_tensor);

	//no way to overload normal setter
	//TODO use python AST to generate these functions
	py_tensor.def("set",
	              [](const PyTensor& t, const PyTensor& t2) { T(t).Set(T(t2)); });

	py_tensor.def_property("val", [](const PyTensor& t) { return t; },
	    [](PyTensor& t, const PyTensor& val) { T(t).Set(T(val)); });

	// indexing
	py_tensor.def("__getitem__", [](const PyTensor& t, const PyTensor& t1) {
		Tensors indices;
		indices.push_back(&t1.Get());
		return PyTensor(&t.Get(), indices);
	});
	py_tensor.def("__getitem__", [](const PyTensor& t, py::tuple indices_tuple) {
		Tensors indices = Reverse(TensorsFromTuple(indices_tuple));
		return PyTensor(&t.Get(), indices);
	});

	py_tensor.def("__setitem__",
	              [](const PyTensor& t, const PyTensor& t1, const PyTensor& t2) {
		              Tensors indices;
		              indices.push_back(&t1.Get());
		              Tensor::Store(t.Get(), T(t2), indices);
	              });
	py_tensor.def("__setitem__", [](const PyTensor& t, py::tuple indices_tuple,
	                                const PyTensor& t2) {
		Tensors indices = Reverse(TensorsFromTuple(indices_tuple));
		Tensor::Store(t.Get(), T(t2), indices);
	});

	py_tensor.def("__setitem__", [](const PyTensor& t, const PyTensor& t1, pybind11::none none) {
		//do nothing
	});
	py_tensor.def("__setitem__", [](const PyTensor& t, py::tuple indices_tuple, pybind11::none none) {
		//do nothing
	});

	// transpose
	py_tensor.def("transpose", [](const PyTensor& t, int dim1, int dim2) {
		return PT(Tensor::Transpose(T(t), -dim1-1, -dim2-1));
	}, py::arg("dim1") = -2, py::arg("dim2") = -1, "Transpose the tensor");

	//transpose property 
	py_tensor.def_property_readonly("T", [](const PyTensor& t) {
		return PT(Tensor::Transpose(T(t)));
	});

	py_tensor.def("__str__", [](const PyTensor& t) {
		return GetNodeString(t.Get().node_);
	});
	py_tensor.def("__repr__", [](const PyTensor& t) {
		return GetNodeString(t.Get().node_);
	});

	py_tensor.def("set_debug_name", [](const PyTensor& t, const std::string& name) {
		t.Get().SetDebugName(name);
	});
}

}  // namespace TensorFrost