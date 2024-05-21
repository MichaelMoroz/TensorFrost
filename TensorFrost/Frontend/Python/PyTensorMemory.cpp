#include "Frontend/Python/PyTensor.h"
#include "Frontend/Python/PyTensorMemory.h"

namespace TensorFrost {

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