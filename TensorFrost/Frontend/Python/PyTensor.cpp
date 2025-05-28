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

std::vector<ArgInfo> GetFunctionArguments(const py::function& func) {
    PyObject* fn = func.ptr();
    PyObject* code_obj = PyFunction_GetCode(fn);
    if (!code_obj) {
        throw std::runtime_error("Could not retrieve code object");
    }

    PyObject* varnames = PyObject_GetAttrString(code_obj, "co_varnames");
    if (!varnames || !PyTuple_Check(varnames)) {
        Py_XDECREF(varnames);
        throw std::runtime_error("Could not retrieve varnames or varnames is not a tuple");
    }

    PyObject* argcount_obj = PyObject_GetAttrString(code_obj, "co_argcount");
    if (!argcount_obj) {
        Py_XDECREF(varnames);
        throw std::runtime_error("Could not retrieve argument count");
    }
    int arg_count = PyLong_AsLong(argcount_obj);
    Py_XDECREF(argcount_obj);
    if (PyErr_Occurred()) {
        Py_XDECREF(varnames);
        throw std::runtime_error("Could not retrieve argument count");
    }

    PyObject* annotations = PyObject_GetAttrString(fn, "__annotations__");
    PyObject* defaults = PyObject_GetAttrString(fn, "__defaults__");

    std::vector<ArgInfo> arg_info_list;
    for (int i = 0; i < arg_count; ++i) {
        PyObject* name = PyTuple_GetItem(varnames, i);
        if (!name || !PyUnicode_Check(name)) {
            Py_XDECREF(varnames);
            Py_XDECREF(annotations);
            Py_XDECREF(defaults);
            throw std::runtime_error("Argument name is not a valid Unicode string");
        }
        std::string arg_name = PyUnicode_AsUTF8(name);

        // Get annotation
        PyObject* annotation = annotations ? PyDict_GetItemString(annotations, arg_name.c_str()) : nullptr;

        // Get default value
        PyObject* default_val = (defaults && PyTuple_Check(defaults) && i >= (arg_count - PyTuple_Size(defaults)))
                                ? PyTuple_GetItem(defaults, i - (arg_count - PyTuple_Size(defaults)))
                                : nullptr;

        py::object annotation_obj = py::reinterpret_borrow<py::object>(annotation);
    	py::object default_obj = py::reinterpret_borrow<py::object>(default_val);

		arg_info_list.emplace_back(arg_name, annotation_obj, default_obj);
    }

    Py_XDECREF(varnames);
    Py_XDECREF(annotations);
    Py_XDECREF(defaults);

    return arg_info_list;
}

py::array ListToArray(py::list input_list) {
	// Get the numpy module
	py::module np = py::module::import("numpy");

	// Convert the list to a numpy array
	py::array np_array = np.attr("array")(input_list);

	return np_array;
}

std::string r_op(const std::string& name) { return "__r" + name + "__"; }

std::string l_op(const std::string& name) { return "__" + name + "__"; }

}  // namespace TensorFrost
