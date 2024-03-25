#include "KernelExecutor.h"

namespace TensorFrost {

int global_kernel_id = 0;

int GenerateKernelID() {
	return global_kernel_id++;
}

}  // namespace TensorFrost