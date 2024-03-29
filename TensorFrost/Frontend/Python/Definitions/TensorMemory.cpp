#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

void TensorMemoryDefinition(py::module& m,
                            py::class_<TensorMemory>& py_tensor_mem) {
	// "constructor"
	m.def(
	    "tensor",
	    [](const std::vector<int>& shape, DataType /*type*/) {
		    return global_memory_manager->Allocate(shape);
	    },
	    "Create a TensorMemory with the given shape");

	// "constructor" from numpy array
	m.def(
	    "tensor",
	    [](std::variant<py::array_t<float>, py::array_t<int>> arr) {
			if (std::holds_alternative<py::array_t<float>>(arr))
			{
			    py::array_t<float> arr_f = std::get<py::array_t<float>>(arr);
			    py::buffer_info info = arr_f.request();

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
						}
						else {
							for (indices[dim] = 0; indices[dim] < info.shape[dim];
								++indices[dim]) {
						    iter_dims(dim + 1, indices);
					    }
				    }
			    };

			    // Start the multi-dimensional iteration
			    std::vector<int> start_indices(info.ndim, 0);
			    iter_dims(0, start_indices);

			    // Allocate the memory
			    return global_memory_manager->AllocateWithData(shape, data, DataType::Float);
		    }
			else if (std::holds_alternative<py::array_t<int>>(arr))
			{
				py::array_t<int> arr_i = std::get<py::array_t<int>>(arr);
			    py::buffer_info info = arr_i.request();

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
						}
						else {
							for (indices[dim] = 0; indices[dim] < info.shape[dim];
								++indices[dim]) {
						    iter_dims(dim + 1, indices);
					    }
				    }
			    };

			    // Start the multi-dimensional iteration
			    std::vector<int> start_indices(info.ndim, 0);
			    iter_dims(0, start_indices);

			    // Allocate the memory
			    return global_memory_manager->AllocateWithData(shape, data, DataType::Int);
			}
			else {
			    throw std::runtime_error("Unsupported data type");
			}
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
	    [](const TensorMemory& t) -> std::variant<py::array_t<float>, py::array_t<int>, py::array_t<uint>> {
			if (t.type == DataType::Float)
			{
			    // create the numpy array
			    py::array_t<float> arr(t.GetShape());

			    // copy the data
			    std::vector<uint> data = global_memory_manager->Readback(&t);
			    float* ptr = static_cast<float*>(arr.request().ptr);
			    for (int i = 0; i < data.size(); i++) {
				    ptr[i] = *(reinterpret_cast<float*>(&data[i]));
			    }

			    return arr;
			}
			else if (t.type == DataType::Int)
			{
			    // create the numpy array
			    py::array_t<int> arr(t.GetShape());

			    // copy the data
			    std::vector<uint> data = global_memory_manager->Readback(&t);
			    int* ptr = static_cast<int*>(arr.request().ptr);
				for (int i = 0; i < data.size(); i++) {
				    ptr[i] = *(reinterpret_cast<int*>(&data[i]));
			    }

			    return arr;
			}
			else if (t.type == DataType::Uint)
			{
			    // create the numpy array
			    py::array_t<uint> arr(t.GetShape());

				// copy the data
				std::vector<uint> data = global_memory_manager->Readback(&t);
			    uint* ptr = static_cast<uint*>(arr.request().ptr);

				for (int i = 0; i < data.size(); i++) {
				    ptr[i] = data[i];
			    }

				return arr;
			}
			else {
			    throw std::runtime_error("Unsupported data type");
			}
	    },
	    "Readback data from tensor memory to a numpy array");

	m.def(
	    "used_memory", []() { return global_memory_manager->GetAllocatedSize(); },
	    "Get the amount of memory currently used by the memory manager");

	m.def("show_window", [](int width, int height, string title) {
		ShowWindow(width, height, title.c_str());
		}, "Show the memory manager window");

	m.def("hide_window", []() { HideWindow(); }, "Hide the memory manager window");

	m.def("render_frame", [](const TensorMemory& t) {
		RenderFrame(t);
		}, "Render a frame from the tensor memory");

	m.def("window_should_close", []() { return WindowShouldClose(); },
	    "Check if the window should close");

	m.def("get_mouse_position", []() { return GetMousePosition(); },
	    "Get the current mouse position");
}

TensorMemory::~TensorMemory() { manager->Free(this); }

}  // namespace TensorFrost