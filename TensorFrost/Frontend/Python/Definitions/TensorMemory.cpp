#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

void TensorMemoryDefinition(py::module& m,
                            py::class_<TensorMemory>& py_tensor_mem) {
	// "constructor"
	m.def(
	    "memory",
	    [](const std::vector<int>& shape, DataType /*type*/) {
		    return global_memory_manager->Allocate(shape);
	    },
	    "Create a TensorMemory with the given shape");

	// "constructor" from numpy array
	m.def(
	    "memory",
	    [](const py::array_t<float>& arr) {
		    // get the shape
		    std::vector<int> shape;
		    py::buffer_info info = arr.request();
		    int size = 1;
		    shape.resize(info.ndim);
		    for (int i = 0; i < info.ndim; i++) 
			{
			    shape[i] = (int)info.shape[i];
			    size *= shape[i];
		    }

		    // create the data vector
		    std::vector<uint> data;

		    // copy the data
		    auto* ptr = static_cast<float*>(info.ptr);
		    data.reserve(size);
		    for (int i = 0; i < size; i++) {
			    data.push_back(*(reinterpret_cast<uint*>(&ptr[i])));
		    }

		    // allocate the memory
		    return global_memory_manager->AllocateWithData(shape, data);
	    },
	    "Create a TensorMemory from a numpy array");

	// properties
	py_tensor_mem.def_property_readonly("shape", [](const TensorMemory& t) {
		std::vector<int> shape = t.GetShape();
		return py::make_tuple(shape);
	});

	// to numpy array
	py_tensor_mem.def_property_readonly(
	    "numpy",
	    [](const TensorMemory& t) {
		    // get the shape
		    std::vector<int> shape = t.GetShape();
			
			std::vector<int> shape2;
			shape2.resize(shape.size());
			for (int i = 0; i < shape.size(); i++)
			{
				shape2[i] = shape[shape.size() - i - 1];
			}

		    // create the numpy array
			py::array_t<float> arr(shape2);

		    // copy the data
		    std::vector<uint> data = global_memory_manager->Readback(&t);
		    float* ptr = static_cast<float*>(arr.request().ptr);
		    for (int i = 0; i < data.size(); i++) {
			    ptr[i] = *(reinterpret_cast<float*>(&data[i]));
		    }

		    return arr;
	    },
	    "Readback data from tensor memory to a numpy array");

	m.def(
	    "used_memory", []() { return global_memory_manager->GetAllocatedSize(); },
	    "Get the amount of memory currently used by the memory manager");
}

}  // namespace TensorFrost