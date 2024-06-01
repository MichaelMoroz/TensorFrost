#include "TensorMemory.h"

namespace TensorFrost {

size_t GetLinearSize(const vector<size_t>& shape) {
	size_t size = 1;
	for (size_t dim : shape) {
		size *= dim;
	}
	return size;
}

vector<size_t> GetShape(const TFTensor *tensor) {
	vector<size_t> shape;
	for (size_t i = 0; i < tensor->dim; i++) {
		shape.push_back(tensor->shape[i]);
	}
	return shape;
}

size_t GetSize(const TFTensor *tensor) {
	size_t size = 1;
	for (size_t i = 0; i < tensor->dim; i++) {
		size *= tensor->shape[i];
	}
	return size;
}

TensorMemoryManager* global_memory_manager = nullptr;

}  // namespace TensorFrost