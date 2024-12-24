#include "Frontend/Python/PyTensor.h"
#include "Frontend/Python/PyTensorMemory.h"

namespace TensorFrost {
PyTensorMemory::PyTensorMemory(py::array arr) {
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
    TFDataFormat format = TFTypeNone;
    switch (info.format[0]) {
        case 'f': // float32
            convert = [](char* ptr) { return *reinterpret_cast<uint32_t*>(ptr); };
            format = TFTypeFloat32;
            break;
        case 'i': // int32
            convert = [](char* ptr) { return static_cast<uint32_t>(*reinterpret_cast<int32_t*>(ptr)); };
            format = TFTypeInt32;
            break;
    		case 'q': // int64 (convert to int32 before casting)
    				convert = [](char* ptr) {
    						int32_t val = (int32_t)*reinterpret_cast<int64_t*>(ptr);
    						return *reinterpret_cast<uint32_t*>(&val);
    				};
    				format = TFTypeInt32;
    				break;
    	  case 'Q': // uint64 (convert to uint32 before casting)
    					convert = [](char* ptr) {
    						uint32_t val = (uint32_t)*reinterpret_cast<uint64_t*>(ptr);
    						return val;
    				};
    				format = TFTypeUint32;
    				break;
        case 'L': // uint32
        case 'I': // uint32
            convert = [](char* ptr) { return *reinterpret_cast<uint32_t*>(ptr); };
            format = TFTypeUint32;
            break;
        case '?': // bool
            convert = [](char* ptr) { return static_cast<uint32_t>(*reinterpret_cast<bool*>(ptr)); };
            format = TFTypeBool32;
            break;
        case 'd': // float64 (convert to float32 before casting)
            convert = [](char* ptr) { float val = (float)*reinterpret_cast<double*>(ptr); return *reinterpret_cast<uint32_t*>(&val); };
            format = TFTypeFloat32;
            break;
        case 'l': // int64 (convert to int32 before casting)
            convert = [](char* ptr) { int32_t val = (int32_t)*reinterpret_cast<int64_t*>(ptr); return *reinterpret_cast<uint32_t*>(&val); };
            format = TFTypeInt32;
            break;
        default:
            throw std::runtime_error("Unsupported data type to create TensorMemory from numpy array, format: " + std::string(info.format));
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

    // Start the multidimensional iteration
    std::vector<size_t> start_indices(info.ndim, 0);
    iter_dims(0, start_indices);

    // Allocate the memory
    tensor_ = global_memory_manager->AllocateTensorWithData(shape, data, format);
}

vector<PyTensorMemory*> TensorMemoryFromTuple(const py::tuple& tuple) {
    vector<PyTensorMemory*> memories;
    for (auto arg : tuple) {
        memories.push_back(&arg.cast<PyTensorMemory&>());
    }
    return memories;
}

vector<PyTensorMemory*> TensorMemoryFromList(const py::list& list) {
    vector<PyTensorMemory*> memories;
    for (auto arg : list) {
        memories.push_back(&arg.cast<PyTensorMemory&>());
    }
    return memories;
}

}  // namespace TensorFrost