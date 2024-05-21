#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>
#include <Frontend/Python/PyTensorMemory.h>

namespace TensorFrost {

void TensorMemoryDefinition(py::module& m,
                            py::class_<PyTensorMemory>& py_tensor_mem) {
	//define constructors from numpy arrays
	py_tensor_mem.def(py::init([](py::array_t<float> arr) {
		return PyTensorMemory(arr, DataType::Float);
	}), "Create a TensorMemory from a numpy array", py::return_value_policy::take_ownership);

	py_tensor_mem.def(py::init([](py::array_t<int> arr) {
		return PyTensorMemory(arr, DataType::Int);
	}), "Create a TensorMemory from a numpy array", py::return_value_policy::take_ownership);

	py_tensor_mem.def(py::init([](py::array_t<uint> arr) {
		return PyTensorMemory(arr, DataType::Uint);
	}), "Create a TensorMemory from a numpy array", py::return_value_policy::take_ownership);

	py_tensor_mem.def(py::init([](py::array_t<bool> arr) {
		return PyTensorMemory(arr, DataType::Bool);
	}), "Create a TensorMemory from a numpy array", py::return_value_policy::take_ownership);

	// "constructor"
	m.def(
	    "tensor",
	    [](const std::vector<int>& shape, DataType type) {
		    return PyTensorMemory(shape, type);
	    },"Create a TensorMemory with the given shape", py::return_value_policy::take_ownership);

	// "constructor" from numpy array
	m.def(
	    "tensor",
	    [](std::variant<py::array_t<float>, py::array_t<int>, py::array_t<uint>, py::array_t<bool>> arr) {
		    if (std::holds_alternative<py::array_t<float>>(arr)) {
		    	return new PyTensorMemory(std::get<py::array_t<float>>(arr), DataType::Float);
		    } else if (std::holds_alternative<py::array_t<int>>(arr)) {
				return new PyTensorMemory(std::get<py::array_t<int>>(arr), DataType::Int);
		    } else if (std::holds_alternative<py::array_t<uint>>(arr)) {
			    return new PyTensorMemory(std::get<py::array_t<uint>>(arr), DataType::Uint);
		    } else if (std::holds_alternative<py::array_t<bool>>(arr)) {
			    return new PyTensorMemory(std::get<py::array_t<bool>>(arr), DataType::Bool);
		    } else {
			    throw std::runtime_error("Unsupported data type");
		    }
	    },
	    "Create a TensorMemory from a numpy array", py::return_value_policy::take_ownership);

	// properties
	py_tensor_mem.def_property_readonly("shape", [](const TensorProp& t) {
		return py::make_tuple(GetShape(&t));
	});

	// to numpy array
	py_tensor_mem.def_property_readonly(
	    "numpy",
	    [](const PyTensorMemory& t)
	        -> std::variant<py::array_t<float>, py::array_t<int>,
	                        py::array_t<uint>, py::array_t<bool>> {
		    if (t.GetType() == DataType::Float) {
		    	return t.ToPyArray<float>();
		    } else if (t.GetType() == DataType::Int) {
		    	return t.ToPyArray<int>();
		    } else if (t.GetType() == DataType::Uint) {
			    return t.ToPyArray<uint>();
		    } else if (t.GetType() == DataType::Bool) {
			    return t.ToPyArray<bool>();
		    } else {
			    throw std::runtime_error("Unsupported data type");
		    }
	    },
	    "Readback data from tensor memory to a numpy array", py::return_value_policy::take_ownership);

	m.def("allocated_memory", []() { return global_memory_manager->GetAllocatedSize(); },
	    "Get the amount of memory currently used by the memory manager");

	m.def("unused_allocated_memory", []() { return global_memory_manager->GetUnusedAllocatedSize(); },
	    "Get the amount of memory currently allocated but not used by the memory manager");
}

}  // namespace TensorFrost