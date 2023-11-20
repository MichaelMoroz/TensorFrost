#include "TensorMemory.h"

namespace TensorFrost {

int GetLinearSize(const vector<int>& shape) {
	int size = 1;
	for (int dim : shape) {
		size *= dim;
	}
	return size;
}

}// namespace TensorFrost