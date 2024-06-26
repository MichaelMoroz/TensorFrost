#include "Frontend/Python/PyTensor.h"

#include "PyTensorMemory.h"

namespace TensorFrost {

PyTensors PyTensorsFromTuple(const py::tuple& tuple) {
	PyTensors tensors;
	for (auto arg : tuple) {
		tensors.push_back(&arg.cast<PyTensor&>());
	}
	return tensors;
}

Tensors TensorsFromTuple(const py::tuple& tuple) {
	Tensors tensors;
	for (auto arg : tuple) {
		tensors.push_back(&arg.cast<PyTensor&>().Get());
	}
	return tensors;
}

tuple<PyTensor*, PyTensor*, PyTensor*> SliceToTensors(const py::slice& slice) {
	PyObject* pyslice = slice.ptr();
	PySliceObject* slice_obj = (PySliceObject*)pyslice;
	PyObject* start = slice_obj->start;
	PyObject* stop = slice_obj->stop;
	PyObject* step = slice_obj->step;

	py::object start_obj = py::reinterpret_borrow<py::object>(start);
	py::object stop_obj = py::reinterpret_borrow<py::object>(stop);
	py::object step_obj = py::reinterpret_borrow<py::object>(step);

	PyTensor* start_tensor = nullptr;
	PyTensor* stop_tensor = nullptr;
	PyTensor* step_tensor = nullptr;

	start_tensor = &start_obj.cast<PyTensor&>();
	stop_tensor = &stop_obj.cast<PyTensor&>();
	step_tensor = &step_obj.cast<PyTensor&>();

	return {start_tensor, stop_tensor, step_tensor};
}

//Tensors TensorsFromTensorIndices(const Tensor* t, const py::tuple& tuple) {
//	Tensors tensors;
//	for (auto arg : tuple) {
//		//if index is a tensor
//		if (py::isinstance<PyTensor>(arg)) {
//			tensors.push_back(&arg.cast<PyTensor&>().Get());
//		} // if index is a slice
//		else if (py::isinstance<py::slice>(arg)) {
//			auto slice = arg.cast<py::slice>();
//			//get native python slice
//			PyObject* pyslice = slice.ptr();
//			PySliceObject* slice_obj = (PySliceObject*)pyslice;
//			//get start, stop, and step
//			PyObject* start = slice_obj->start;
//			PyObject* stop = slice_obj->stop;
//			PyObject* step = slice_obj->step;
//
//			//convert start, stop, and step to py::object
//			py::object start_obj = py::reinterpret_borrow<py::object>(start);
//			py::object stop_obj = py::reinterpret_borrow<py::object>(stop);
//			py::object step_obj = py::reinterpret_borrow<py::object>(step);
//
//			//try to cast to PyTensor
//			PyTensor* start_tensor = nullptr;
//			PyTensor* stop_tensor = nullptr;
//			PyTensor* step_tensor = nullptr;
//
//			try {
//				start_tensor = &start_obj.cast<PyTensor&>();
//			} catch (const py::cast_error& e) {
//				//do nothing
//			}
//
//			try {
//				stop_tensor = &stop_obj.cast<PyTensor&>();
//			} catch (const py::cast_error& e) {
//				//do nothing
//			}
//
//			try {
//				step_tensor = &step_obj.cast<PyTensor&>();
//			} catch (const py::cast_error& e) {
//				//do nothing
//			}
//
//			//if start, stop, and step are all PyTensor
//
//
//
//		    
//
//		else {
//			throw std::invalid_argument("Invalid index type");
//		}
//	}
//	return tensors;
//}

PyTensors PyTensorsFromList(const py::list& list) {
	PyTensors tensors;
	for (auto arg : list) {
		tensors.push_back(&arg.cast<PyTensor&>());
	}
	return tensors;
}

Tensors TensorsFromList(const py::list& list) {
	Tensors tensors;
	for (auto arg : list) {
		tensors.push_back(&arg.cast<PyTensor&>().Get());
	}
	return tensors;
}

PyTensors PyTensorsFromTensors(const Tensors& tensors) {
	PyTensors py_tensors;
	for (const auto* tensor : tensors) {
		py_tensors.push_back(new PyTensor(tensor));
	}
	return py_tensors;
}

std::variant<PyTensor *, py::tuple> PyTensorsToTupleVariant(const PyTensors &tensors) {
	if (tensors.size() == 1) {
		//if there is only one tensor, return the tensor
		return tensors[0];
	} else {
		//convert to py::tuple of PyTensor*
		return py::tuple(py::cast(tensors));
	}
}

void UpdateTensorNames() {
	PyObject* p = PyEval_GetLocals();
	py::dict all_names = py::reinterpret_borrow<py::dict>(p ? p : py::module_::import("__main__").attr("__dict__").ptr());
	
	for (auto item : all_names) {
		std::string var_name = py::str(item.first);
		py::object var_value = py::reinterpret_borrow<py::object>(item.second);
		if (py::isinstance<PyTensor>(var_value)) {
			PyTensor& py_tensor = var_value.cast<PyTensor&>();
			const Tensor* tensor = &py_tensor.Get();
			tensor->SetDebugName(var_name);
		}
	}
}

std::string r_op(const std::string& name) { return "__r" + name + "__"; }

std::string l_op(const std::string& name) { return "__" + name + "__"; }

}  // namespace TensorFrost
