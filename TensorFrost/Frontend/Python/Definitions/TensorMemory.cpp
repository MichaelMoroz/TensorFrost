#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>
#include <Frontend/Python/PyTensorMemory.h>

namespace TensorFrost {

void TensorMemoryDefinition(py::module& m,
                            py::class_<PyTensorMemory>& py_tensor_mem) {
	//define constructors from numpy arrays
	py_tensor_mem.def(py::init([](py::array arr) {
		return PyTensorMemory(arr);
	}), "Create a TensorMemory from a numpy array", py::return_value_policy::take_ownership);

	// "constructor"
	m.def(
	    "tensor",
	    [](const std::vector<size_t>& shape, TFType type) {
		    return PyTensorMemory(shape, type);
	    },"Create a TensorMemory with the given shape", py::return_value_policy::take_ownership);

	// "constructor" from numpy array
	m.def(
	    "tensor",
	    [](py::array arr) {
	    	return new PyTensorMemory(arr);
	    },
	    "Create a TensorMemory from a numpy array", py::return_value_policy::take_ownership);

	// properties
	py_tensor_mem.def_property_readonly("shape", [](const PyTensorMemory& t) {
		vector<size_t> shape = t.Shape();
		return py::cast(shape);
	});

	py_tensor_mem.def_property_readonly("type", [](const PyTensorMemory& t) {
		return t.GetType();
	});

	py_tensor_mem.def_property_readonly("size", [](const PyTensorMemory& t) {
		return GetSize(t.tensor_);
	});

	// to numpy array
	py_tensor_mem.def_property_readonly(
	    "numpy",
	    [](const PyTensorMemory& t)
	        -> std::variant<py::array_t<float>, py::array_t<int>,
	                        py::array_t<uint>, py::array_t<bool>> {
		    if (t.GetType() == TFType::Float) {
		    	return t.ToPyArray<float>();
		    } else if (t.GetType() == TFType::Int) {
		    	return t.ToPyArray<int>();
		    } else if (t.GetType() == TFType::Uint) {
			    return t.ToPyArray<uint>();
		    } else if (t.GetType() == TFType::Bool) {
			    return t.ToPyArray<bool>();
		    } else {
			    throw std::runtime_error("Unsupported data type for numpy conversion");
		    }
	    },
	    "Readback data from tensor memory to a numpy array", py::return_value_policy::take_ownership);

	m.def("allocated_memory", []() { return global_memory_manager->GetAllocatedSize(); },
	    "Get the amount of memory currently used by the memory manager");

	m.def("unused_allocated_memory", []() { return global_memory_manager->GetUnusedAllocatedSize(); },
	    "Get the amount of memory currently allocated but not used by the memory manager");
}

}  // namespace TensorFrost