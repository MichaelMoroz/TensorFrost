#include "Backend.h"

namespace TensorFrost {

TensorMemoryManager* GlobalMemoryManager = nullptr;

void InitializeMemoryManager(/*BackendType backendType*/) {
	GlobalMemoryManager = new CPU_VM_MemoryManager();
}

}// namespace TensorFrost