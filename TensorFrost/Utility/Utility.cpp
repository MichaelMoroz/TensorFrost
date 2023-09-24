#include "Utility.h"

namespace TensorFrost {

int GetSize(vector<int> shape) {
    int size = 1;
    for (int i = 0; i < shape.size(); i++) {
        size *= shape[i];
    }
    return size;
}

}  // namespace TensorFrost