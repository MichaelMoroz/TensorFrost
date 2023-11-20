#include "TensorProgram.h"

namespace TensorFrost {

void TensorProgram::CreateExecutionGraph()
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
	Tensor::SetEvaluationContext(nullptr);
}

}