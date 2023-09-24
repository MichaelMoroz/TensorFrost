#include "Utility.h"

namespace TensorFrost {

int GetSize(const vector<int>& shape) {
	int size = 1;
	for (int i : shape) {
		size *= i;
	}
	return size;
}

}  // namespace TensorFrost