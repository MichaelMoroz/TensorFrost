#include <utility>
#include <vector>

#include <TensorFrost.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace TensorFrost {

// Tensor wrapper for python
class PyTensor {
	Tensor* tensor_;

 public:
	explicit PyTensor(Tensor* tensor) : tensor_(tensor) {}
	~PyTensor() = default;

	[[nodiscard]] Tensor& Get() const { return *tensor_; }

	PyTensor(const std::vector<int>& shape, DataType type = DataType::Float) {
		switch (type) {
			case DataType::Float:
				tensor_ = &Tensor::Constant(shape, 0.0F);
				break;
			case DataType::Int:
				tensor_ = &Tensor::Constant(shape, 0);
				break;
			case DataType::Uint:
				tensor_ = &Tensor::Constant(shape, 0U);
				break;
			default:
				throw std::runtime_error("Invalid data type");
		}
	}

	PyTensor(const TensorView& indexed_tensor) {
		// load the elements of the indexed tensor
		tensor_ = &Tensor::Load(*indexed_tensor.value, indexed_tensor.indices);
	}

	PyTensor(float value) { tensor_ = &Tensor::Constant(Shape(), value); }
	PyTensor(int value) { tensor_ = &Tensor::Constant(Shape(), value); }
	PyTensor(unsigned int value) { tensor_ = &Tensor::Constant(Shape(), value); }
};

Tensor TensorFromPyArray(const py::array_t<float>& array) {
	auto buffer = array.request();
	auto* ptr = static_cast<float*>(buffer.ptr);
	std::vector<int> shape = std::vector<int>();
	for (int i = 0; i < buffer.ndim; i++) {
		shape.push_back(buffer.shape[i]);
	}
	return Tensor::Constant(shape, ptr);
}

py::array_t<float> TensorToPyArray(const Tensor& tensor) {
	std::vector<int> shape = tensor.shape.GetShape();
	py::array::ShapeContainer shape2 =
	    py::array::ShapeContainer(shape.begin(), shape.end());
	py::array_t<float> array(shape2);
	auto buffer = array.request();
	auto* ptr = static_cast<float*>(buffer.ptr);
	for (int i = 0; i < tensor.Size(); i++) {
		ptr[i] = 0.0;
	}
	return array;
}

PYBIND11_MODULE(TensorFrost, m) {
	//py::enum_<TensorFrost::DataType>(m, "DataType")
	//    .value("float", TensorFrost::DataType::Float)
	//    .value("int", TensorFrost::DataType::Int)
	//    .value("uint", TensorFrost::DataType::Uint)
	//    .value("bool", TensorFrost::DataType::Bool);

#define PT(tensor) PyTensor(&(tensor))
#define T(tensor) (tensor).Get()

	auto pyTensor = py::class_<PyTensor>(m, "Tensor");

	// initializers
	pyTensor.def(py::init<std::vector<int>, DataType>())
	    .def(py::init<const TensorView&>())
	    .def(py::init<float>())
	    .def(py::init<int>())
	    .def(py::init<unsigned int>());

	// properties
	pyTensor
	    .def_property_readonly(
	        "shape", [](const PyTensor& t) { return t.Get().shape.GetShape(); })
	    .def_property_readonly("type",
	                           [](const PyTensor& t) { return t.Get().type; })
	    .def("numpy", [](const PyTensor& t) { return TensorToPyArray(t.Get()); })
	    .def("index",
	         [](const PyTensor& t, int dim) { return PT(T(t).Index(dim)); });

	// getter and setter
	pyTensor
	    .def("__getitem__",
	         [](const PyTensor& t, py::tuple indices_tuple) {
		         std::vector<const Tensor*> indices;
		         for (auto arg : indices_tuple) {
			         indices.push_back(&arg.cast<const PyTensor&>().Get());
		         }
		         return TensorView(&t.Get(), indices);
	         })
	    .def("__setitem__",
	         [](const PyTensor& t, py::tuple indices_tuple, const PyTensor& t2) {
		         std::vector<const Tensor*> indices;
		         for (auto arg : indices_tuple) {
			         indices.push_back(&arg.cast<const PyTensor&>().Get());
		         }
		         Tensor::Store(t.Get(), T(t2), indices);
	         });
#define DEFINE_OPERATOR(opname, op)                                           \
	pyTensor.def("__" #opname "__", [](const PyTensor& t, const PyTensor& t2) { \
		return PT(T(t) op T(t2));                                                 \
	});                                                                         \
	pyTensor.def("__r" #opname "__", [](const PyTensor& t, const float f) {     \
		return PT(Tensor::Constant(T(t).shape, f) op T(t));                       \
	});                                                                         \
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
		return PT(Tensor::pow(T(t), Tensor::Constant(T(t).shape, f)));
	});
	pyTensor.def("__pow__", [](const PyTensor& t, py::array_t<float> f) {
		return PT(Tensor::pow(T(t), TensorFromPyArray(f)));
	});
	pyTensor.def("__rpow__", [](const PyTensor& t, float f) {
		return PT(Tensor::pow(Tensor::Constant(T(t).shape, f), T(t)));
	});
	pyTensor.def("__rpow__", [](const PyTensor& t, py::array_t<float> f) {
		return PT(Tensor::pow(TensorFromPyArray(f), T(t)));
	});
	// end power operator
	// end operator overloads
	;

#define DEFINE_OPERATOR(opname, op)                                       \
	.def("__" #opname "__", [](const TensorView& t, const TensorView& t2) { \
		return PT(T(PyTensor(t)) op T(PyTensor(t2)));                         \
	}).def("__" #opname "__", [](const TensorView& t, const float f) {      \
		return PT(Tensor::Constant(t.value->shape, f) op T(PyTensor(t)));     \
	})

	py::class_<TensorView>(m, "TensorView") DEFINE_OPERATOR(add, +)
	    DEFINE_OPERATOR(sub, -) DEFINE_OPERATOR(mul, *) DEFINE_OPERATOR(div, /)
	        DEFINE_OPERATOR(mod, %)
	            // negative
	            .def("__neg__",
	                 [](const TensorView& t) { return PT(-T(PyTensor(t))); })
	    // comparison
	    DEFINE_OPERATOR(eq, ==) DEFINE_OPERATOR(ne, !=) DEFINE_OPERATOR(lt, <)
	        DEFINE_OPERATOR(le, <=) DEFINE_OPERATOR(gt, >) DEFINE_OPERATOR(ge, >=)
	    // logical
	    DEFINE_OPERATOR(and, &&) DEFINE_OPERATOR(or, ||)
	            .def("__not__",
	                 [](const TensorView& t) { return PT(!T(PyTensor(t))); })
	    // bitwise
	    DEFINE_OPERATOR(xor, ^) DEFINE_OPERATOR(lshift, <<)
	        DEFINE_OPERATOR(rshift, >>) DEFINE_OPERATOR(and_, &)
	            DEFINE_OPERATOR(or_, |)
	            .def("__invert__",
	                 [](const TensorView& t) { return PT(~T(PyTensor(t))); })

	            //** operator overload
	            .def("__pow__", [](const TensorView& t, const TensorView& t2) {
		            return PT(Tensor::pow(PyTensor(t).Get(), PyTensor(t2).Get()));
	            });

	// implicit conversion from TensorView to PyTensor
	py::implicitly_convertible<TensorView, PyTensor>();
	py::implicitly_convertible<float, PyTensor>();
	py::implicitly_convertible<int, PyTensor>();
	py::implicitly_convertible<unsigned int, PyTensor>();

// unary functions
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

	m.def("zeros", [](std::vector<int> shape) {
		std::string debug = "Received shape: " + std::to_string(shape[0]);
		for (int i = 1; i < shape.size(); i++) {
			debug += ", " + std::to_string(shape[i]);
		}
		py::print(debug);
		return PT(Tensor::Constant(shape, 0.0F));
	});

	m.def("input",
	      [](std::vector<int> shape) { return PT(Tensor::Input(Shape(shape))); });
	m.def("index", [](int dim, std::vector<int> shape) {
		return PT(Tensor::Index(Shape(shape), dim));
	});
	m.def(
	    "Program",
	    [](const py::function& py_evaluate) {
		    return TensorProgram([py_evaluate]() -> std::vector<Tensor> {
			    py::gil_scoped_acquire acquire;  // Acquire the GIL

			    // 2. Call the Python function
			    py::object result = py_evaluate();

			    // Debug print to check the result
			    py::print("Result from Python function:", result);

			    // 3. Convert back to std::vector<Tensor>
			    auto py_outputs = py::cast<std::vector<PyTensor>>(result);
			    std::vector<Tensor> outputs = std::vector<Tensor>();
			    for (PyTensor output : py_outputs) {
				    outputs.push_back(output.Get());
			    }

			    return outputs;
		    });
	    },
	    "Compile a TensorProgram from a python function");

	py::class_<TensorProgram>(m, "TensorProgram")
	    .def(
	        "__call__",
	        [](TensorProgram& /*program*/, py::list py_inputs) {
		        auto inputs = py::cast<std::vector<PyTensor>>(py_inputs);
		        py::print("Received inputs: " + std::to_string(inputs.size()));
		        std::vector<Tensor> inputs2 = std::vector<Tensor>();
		        for (PyTensor input : inputs) {
			        inputs2.push_back(input.Get());
		        }
		        std::vector<Tensor> outputs =
		            TensorFrost::TensorProgram::Evaluate(inputs2);
		        std::vector<PyTensor> outputs2 = std::vector<PyTensor>();
		        for (Tensor output : outputs) {
			        outputs2.push_back(PT(output));
		        }
		        py::print("Returning outputs: " + std::to_string(outputs2.size()));
		        return py::cast(outputs2);
	        },
	        "Evaluate the TensorProgram with the given inputs")
	    .def("ListGraphOperations", [](TensorProgram& program) {
		    std::string listing = "List of operations:\n";
		    listing += program.ir.GetOperationListing();
		    py::str result = py::str(listing);
		    py::print(result);
	    });
	;

	py::print("TensorFrost module loaded!");
}

}  // namespace TensorFrost