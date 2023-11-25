#include "TensorProgram.h"

namespace TensorFrost {

void TensorProgram::CreateProgram() {
	// create new IR graph
	Tensor::SetEvaluationContext(&ir);
	Tensors outputs = evaluate_callback();
	// set outputs
	for (int i = 0; i < outputs.size(); i++) {
		outputs[i]->SetMemoryType(MemoryType::Output, i);
	}

	ir.Clusterize();
	ir.OptimizeClusters();
	ir.RemoveUnusedNodes();
	ir.PostProcessClusters();
	ir.TransformToLinearIndex();

	program = GenerateProgram(&ir);

	Tensor::SetEvaluationContext(nullptr);

	// compile and load kernel
	CompileAndLoadKernel(program);
}

vector<TensorMemory*> TensorProgram::Evaluate(
    const vector<TensorMemory*>& input) const {
	return ExecuteProgram(program, input);
}

}  // namespace TensorFrost