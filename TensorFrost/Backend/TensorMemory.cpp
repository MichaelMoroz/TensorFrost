#include "TensorMemory.h"

namespace TensorFrost {

int GetLinearSize(const vector<int>& shape) {
	int size = 1;
	for (int dim : shape) {
		size *= dim;
	}
	return size;
}

vector<int> GetShape(const TF_Tensor *tensor) {
	vector<int> shape;
	for (uint i = 0; i < tensor->dim; i++) {
		shape.push_back(tensor->shape[i]);
	}
	return shape;
}

int GetSize(const TF_Tensor *tensor) {
	int size = 1;
	for (uint i = 0; i < tensor->dim; i++) {
		size *= tensor->shape[i];
	}
	return size;
}

TensorMemoryManager* global_memory_manager = nullptr;

}  // namespace TensorFrost