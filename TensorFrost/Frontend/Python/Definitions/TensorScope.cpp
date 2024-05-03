#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

void ScopeDefinitions(py::module& m, py::class_<PyTensor>& py_tensor) {
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

	m.def(
	    "if_cond",
	    [](const PyTensor& condition, const py::function& true_body) {
		    std::function<void()> f = [&true_body]() {
			    py::gil_scoped_acquire acquire;
			    true_body();
		    };
		    Tensor::If(T(condition), f);
	    },
	    py::arg("condition"), py::arg("true_body"));

	m.def(
	    "if_cond",
	    [](const PyTensor& condition, const py::function& true_body,
	       const py::function& false_body) {
		    std::function<void()> f1 = [&true_body]() {
			    py::gil_scoped_acquire acquire;
			    true_body();
		    };
		    std::function<void()> f2 = [&false_body]() {
			    py::gil_scoped_acquire acquire;
			    false_body();
		    };
		    Tensor::If(T(condition), f1, f2);
	    },
	    py::arg("condition"), py::arg("true_body"), py::arg("false_body"));

	m.def("break_loop", []() { Tensor::Break(); });
	m.def("continue_loop", []() { Tensor::Continue(); });

	m.def(
	    "kernel",
	    [](py::list shape, const py::function& body) {
		    // wrap the function to convert the PyTensor to Tensor
		    std::function<void(const vector<Tensor*>&)> f2 =
		        [&body](const vector<Tensor*>& tensors) {
			        py::gil_scoped_acquire acquire;
			        PyTensors py_tensors = PyTensorsFromVector(tensors);
			        body(py_tensors);
		        };

		    Tensors shape_tensors = TensorsFromList(shape);

		    Tensor::Kernel(shape_tensors, f2);
	    },
	    py::arg("shape"), py::arg("body"));

	py_tensor.def("__enter__", &PyTensor::__enter__);
	py_tensor.def("__exit__", &PyTensor::__exit__);

	//loop scope
	m.def("loop",
	[](const PyTensor& begin, const PyTensor& end, const PyTensor& step) {
		Tensor& for_loop = Tensor::Loop(T(begin), T(end), T(step));
		return PT(for_loop);
	});

	m.def("loop",
	[](const PyTensor& begin, const PyTensor& end) {
		Tensor& for_loop = Tensor::Loop(T(begin), T(end), T(PyTensor(1)));
		return PT(for_loop);
	});

	m.def("loop",
	[](const PyTensor& end) {
		Tensor& for_loop = Tensor::Loop(T(PyTensor(0)), T(end), T(PyTensor(1)));
		return PT(for_loop);
	});
	
	//if scope
	m.def("if_cond", 
	[](const PyTensor& condition) {
		Tensor& if_cond = Tensor::If(T(condition));
		return PT(if_cond);
	});

	//kernel scope
	m.def("kernel", 
	[](py::list shape) {
		Tensors shape_tensors = TensorsFromList(shape);
		Tensor& kernel = Tensor::Kernel(shape_tensors);
		return PT(kernel);
	});
}

}  // namespace TensorFrost