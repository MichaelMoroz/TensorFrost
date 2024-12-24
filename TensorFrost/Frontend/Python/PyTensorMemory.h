#include <TensorFrost.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace TensorFrost {

namespace py = pybind11;

// Tensor wrapper for python
class PyTensorMemory {
 public:
	TFTensor* tensor_;

	explicit PyTensorMemory(TFTensor* tensor) : tensor_(tensor) {}

	PyTensorMemory(vector<size_t> shape, TFDataFormat type = TFTypeFloat32) {
		tensor_ = global_memory_manager->AllocateTensor(shape, type);
	}

	TFDataFormat GetFormat() const {
		return tensor_->format;
	}

	PyTensorMemory(py::array arr);

	template <typename T>
	py::array_t<T> ToPyArray() const {
		// Get the shape
		std::vector<size_t> shape = GetShape(tensor_);

		// Create the numpy array
		py::array_t<T> arr(shape);

		// Copy the data
		std::vector<uint> data = global_memory_manager->Readback(tensor_);
		T* ptr = static_cast<T*>(arr.request().ptr);
		for (int i = 0; i < data.size(); i++) {
			ptr[i] = *(reinterpret_cast<T*>(&data[i]));
		}

		return arr;
	}

	vector<size_t> Shape() const {
		return TensorFrost::GetShape(tensor_);
	}

	~PyTensorMemory() {
		global_memory_manager->DeallocateTensor(*tensor_);
	}

};

vector<PyTensorMemory*> TensorMemoryFromTuple(const py::tuple& tuple);
vector<PyTensorMemory*> TensorMemoryFromList(const py::list& list);

}  // namespace TensorFrost