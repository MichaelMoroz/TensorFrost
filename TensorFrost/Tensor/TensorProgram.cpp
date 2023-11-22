#include "TensorProgram.h"

namespace TensorFrost {

void TensorProgram::CreateProgram()
{
	// create new IR graph
	Tensor::SetEvaluationContext(&ir);
	Tensors outputs = evaluate_callback();
	// set outputs
	for (const Tensor* output : outputs) {
		output->SetMemoryType(MemoryType::Output);
	}

	ir.Clusterize();
	ir.OptimizeClusters();
	ir.RemoveUnusedNodes();
	ir.PostProcessClusters();
	ir.TransformToLinearIndex();

	program = GenerateProgram(&ir);

	Tensor::SetEvaluationContext(nullptr);

	loadLibraryWin();
}

vector<TensorMemory*> TensorProgram::Evaluate(
    const vector<TensorMemory*> input) {

	return ExecuteProgram(program, input);
}

}