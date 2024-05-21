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
	TensorProp* tensor_;

	explicit PyTensorMemory(TensorProp* tensor) : tensor_(tensor) {}

	PyTensorMemory(vector<int> shape, DataType type = DataType::Float) {
		tensor_ = global_memory_manager->Allocate(shape, type);
	}

	DataType GetType() const {
		return tensor_->type;
	}

	template <typename T>
	PyTensorMemory(py::array_t<T> arr, DataType type = DataType::Float) {
		py::buffer_info info = arr.request();

		// Get the shape
		std::vector<int> shape;
		for (int i = 0; i < info.ndim; i++) {
			shape.push_back((int)info.shape[i]);
		}

		// Create the data vector
		std::vector<uint> data;
		data.reserve(info.size);

		// Define a recursive lambda function for multi-dimensional iteration
		std::function<void(const int, std::vector<int>&)> iter_dims;
		iter_dims = [&iter_dims, &info, &data](const int dim,
											   std::vector<int>& indices) {
			if (dim == info.ndim) {
				// Calculate the actual memory address using strides
				char* ptr = static_cast<char*>(info.ptr);
				for (int i = 0; i < info.ndim; ++i) {
					ptr += indices[i] * info.strides[i];
				}
				data.push_back(*(reinterpret_cast<uint*>(ptr)));
			} else {
				for (indices[dim] = 0; indices[dim] < info.shape[dim]; ++indices[dim]) {
					iter_dims(dim + 1, indices);
				}
			}
		};

		// Start the multi-dimensional iteration
		std::vector<int> start_indices(info.ndim, 0);
		iter_dims(0, start_indices);

		// Allocate the memory
		tensor_ = global_memory_manager->AllocateWithData(shape, data, type);
	}

	template <typename T>
	py::array_t<T> ToPyArray() const {
		// Get the shape
		std::vector<int> shape = GetShape(tensor_);

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

	~PyTensorMemory() {
		global_memory_manager->Free(tensor_);
	}

};

vector<PyTensorMemory*> TensorMemoryFromTuple(const py::tuple& tuple);
vector<PyTensorMemory*> TensorMemoryFromList(const py::list& list);

}  // namespace TensorFrost