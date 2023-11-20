#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

void TensorMemoryDefinition(py::module& m, py::class_<TensorMemory>& py_tensor_mem) {
	// "constructor"
	m.def(
	    "memory",
	    [](const std::vector<int>& shape, DataType type) {
        return GlobalMemoryManager->Allocate(shape);
    }, "Create a TensorMemory with the given shape");
    
    // "constructor" from numpy array
    m.def("memory", [](const py::array_t<float>& arr) {
        // get the shape
        std::vector<int> shape;
        py::buffer_info info = arr.request();
        int size = 1;
        for (int dim : info.shape) {
            shape.push_back(dim);
            size *= dim;
        }

        // create the data vector
        std::vector<uint> data;

        // copy the data
        float* ptr = (float*)info.ptr;
        for (int i = 0; i < size; i++) {
            data.push_back(*((uint*)&ptr[i]));
        }

        // allocate the memory
        return GlobalMemoryManager->AllocateWithData(shape, data);
    }, "Create a TensorMemory from a numpy array");

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

        // create the numpy array
        py::array_t<float> arr(shape);

        // copy the data
        std::vector<uint> data = GlobalMemoryManager->Readback(&t);
        float* ptr = (float*)arr.request().ptr;
        for (int i = 0; i < data.size(); i++) {
	        ptr[i] = *((float*)&data[i]);
        }

        return arr;
    }, "Readback data from tensor memory to a numpy array");

    m.def(
	    "used_memory",
	    []() { return GlobalMemoryManager->GetAllocatedSize(); },
	    "Get the amount of memory currently used by the memory manager");
}

}  // namespace TensorFrost