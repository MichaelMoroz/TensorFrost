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

	PyTensorMemory(vector<size_t> shape, TFType type = TFType::Float) {
		tensor_ = global_memory_manager->AllocateTensor(shape, type);
	}

	TFType GetType() const {
		return tensor_->type;
	}

	PyTensorMemory(py::array arr, TFType type = TFType::Float) {
	    py::buffer_info info = arr.request();

	    // Get the shape
	    std::vector<size_t> shape;
	    for (size_t i = 0; i < (size_t)info.ndim; i++) {
	        shape.push_back(info.shape[i]);
	    }

	    // Create the data vector
	    std::vector<uint32_t> data;
	    data.reserve(info.size);

	    // Determine the data type and conversion function
	    std::function<uint32_t(char*)> convert;
	    switch (info.format[0]) {
	        case 'f': // float32
	            convert = [](char* ptr) { return *reinterpret_cast<uint32_t*>(ptr); };
	            break;
	        case 'i': // int32
	            convert = [](char* ptr) { return static_cast<uint32_t>(*reinterpret_cast<int32_t*>(ptr)); };
	            break;
	        case 'I': // uint32
	            convert = [](char* ptr) { return *reinterpret_cast<uint32_t*>(ptr); };
	            break;
	        case '?': // bool
	            convert = [](char* ptr) { return static_cast<uint32_t>(*reinterpret_cast<bool*>(ptr)); };
	            break;
	    	case 'd': // float64 (convert to float32 before casting)
	            convert = [](char* ptr) { float val = (float)*reinterpret_cast<double*>(ptr); return *reinterpret_cast<uint32_t*>(&val); };
	    		break;
	    	case 'l': // int64 (convert to int32 before casting)
	            convert = [](char* ptr) { int32_t val = (int32_t)*reinterpret_cast<int64_t*>(ptr); return *reinterpret_cast<uint32_t*>(&val); };
	    		break;
	        default:
	            throw std::runtime_error("Unsupported data type");
	    }

	    // Define a recursive lambda function for multi-dimensional iteration
	    std::function<void(const size_t, std::vector<size_t>&)> iter_dims;
	    iter_dims = [&](const size_t dim, std::vector<size_t>& indices) {
	        if (dim == info.ndim) {
	            // Calculate the actual memory address using strides
	            char* ptr = static_cast<char*>(info.ptr);
	            for (size_t i = 0; i < (size_t)info.ndim; ++i) {
	                ptr += indices[i] * info.strides[i];
	            }
	            data.push_back(convert(ptr));
	        } else {
	            for (indices[dim] = 0; indices[dim] < (size_t)info.shape[dim]; ++indices[dim]) {
	                iter_dims(dim + 1, indices);
	            }
	        }
	    };

	    // Start the multi-dimensional iteration
	    std::vector<size_t> start_indices(info.ndim, 0);
	    iter_dims(0, start_indices);

	    // Allocate the memory
	    tensor_ = global_memory_manager->AllocateTensorWithData(shape, data, type);
	}

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

	~PyTensorMemory() {
		global_memory_manager->DeallocateTensor(*tensor_);
	}

};

vector<PyTensorMemory*> TensorMemoryFromTuple(const py::tuple& tuple);
vector<PyTensorMemory*> TensorMemoryFromList(const py::list& list);

}  // namespace TensorFrost