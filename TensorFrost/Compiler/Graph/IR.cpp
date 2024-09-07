#include "IR.h"

namespace TensorFrost {

void IR::RemoveNode(Node* node) {
    if (node->valid()) {
        // if child node exists, iterate through it and remove all children
        if (node->child) {
            vector<Node*> to_delete;
            for (auto child = NodeIterator(node); !child.end(); child.next()) {
                to_delete.push_back(*child);
            }
            for (int i = (int)to_delete.size() - 1; i >= 0; i--) {
                RemoveNode(to_delete[i]);
            }
        }

        // if direct child of its parent
        if (node->parent && node->parent->child == node) {
            node->parent->child = node->next;
        } else if (node->prev) {
            node->prev->next = node->next;
        }

        node->next->prev = node->prev;
        delete node;
    }
}

//#define PROFILE_COMPILATION

void IR::RunCompilationPass(string pass_name, const function<void()>& expression, bool print, bool update_graph) {
	current_pass = pass_name;
#ifdef PROFILE_COMPILATION
	auto start = std::chrono::high_resolution_clock::now();
#endif

	try {
		expression();
	} catch (const std::exception& e) {
		throw std::runtime_error("Error in compilation pass " + pass_name + ": " + e.what());
	}

#ifdef PROFILE_COMPILATION
	auto end = std::chrono::high_resolution_clock::now();
	float duration = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f;

	PassStats stats;
	stats.pass_name = pass_name;
	stats.duration = duration;
	stats.node_count = 0;
	for (auto node = begin(); !node.end(); node.next()) {
		stats.node_count++;
	}
	pass_stats.push_back(stats);
#endif

	if (update_graph) {
		UpdateGraph();
	}

	if (print) {
		CheckIR(pass_name, false, false);
	}
	current_pass = "";
}

#define MAX_COMPILATION_ITERATIONS 32
#define LOAD_FUSION

void IR::RunIterativeCompilationPass(string pass_name, int max_iterations, const function<bool()>& expression, bool print, bool update_graph) {
	RunCompilationPass(pass_name, [&]() {
		bool converged = false;
		for (int i = 0; i < max_iterations; i++) {
			bool finished = expression();
			if (finished) {
				converged = true;
				break;
			}
		}
		if (!converged) {
			throw std::runtime_error("Failed to converge on " + pass_name + ", exceeded maximum iterations");
		}
	}, print, update_graph);
}

void IR::CompileIR()
{
	// TODO (Moroz): Add auto tests into build system
	CheckIR("Input", false, false);
	RunCompilationPass("GetInputList", [&]() { GetInputList(); });
	RunCompilationPass("OptimizeOperations", [&]() { OptimizeOperations(); });
	RunCompilationPass("UnrollLoops", [&]() { UnrollLoops(); }, true);
	RunCompilationPass("TryReplaceModificationsWithVersions", [&]() { TryReplaceModificationsWithVersions(); }, true);
	RunCompilationPass("RemoveUnusedOperations", [&]() { RemoveUnusedOperations(); }, true);

	RunIterativeCompilationPass("InsertAlgorithmicPrimitives_PreAutodiff", MAX_COMPILATION_ITERATIONS, [&]() {
		return InsertAlgorithmicPrimitives(true);
	}, true);

	RunCompilationPass("ComputeAutodiff", [&]() { ComputeAutodiff(); });

	RunCompilationPass("RemoveUnusedOperations", [&]() { RemoveUnusedOperations(); }, true);
	RunCompilationPass("UnrollAtomicOperations", [&]() { UnrollAtomicOperations(); });
	RunCompilationPass("OptimizeReductions", [&]() { OptimizeReductions(); }, true);

	RunIterativeCompilationPass("InsertAlgorithmicPrimitives_PostAutodiff", MAX_COMPILATION_ITERATIONS, [&]() {
		return InsertAlgorithmicPrimitives(false);
	}, true);

	RunCompilationPass("TryReplaceModificationsWithVersions", [&]() { TryReplaceModificationsWithVersions(); });
	RunCompilationPass("OptimizeOperations", [&]() { OptimizeOperations(); });
	RunCompilationPass("RemoveUnusedOperations", [&]() { RemoveUnusedOperations(); }, true);

	RunCompilationPass("SeparateOperationsIntoKernels", [&]() { SeparateOperationsIntoKernels(); }, true);
	RunCompilationPass("CheckKernelShapes", [&]() { CheckKernelShapes(); });
	RunCompilationPass("ReorderOperations", [&]() { ReorderOperations(); });
	RunCompilationPass("MoveShapeOutsideKernels", [&]() { MoveShapeOutsideKernels(); });
	RunCompilationPass("OptimizeKernels", [&]() { OptimizeKernels(); });
	RunCompilationPass("OptimizeHost", [&]() { OptimizeHost(); });

	RunCompilationPass("UnrollLoops", [&]() { UnrollLoops(4); });
	RunCompilationPass("TryReplaceModificationsWithVersions", [&]() { TryReplaceModificationsWithVersions(); }, true);
	RunCompilationPass("RemoveUnusedOperations", [&]() { RemoveUnusedOperations(); });

#ifdef LOAD_FUSION
	RunIterativeCompilationPass("Iterative load fusion", MAX_COMPILATION_ITERATIONS, [&]() {
		AddKernelGlobalLoadOperations();
		AddMemoryOpIndices();
		bool no_changes = OptimizeKernelLoadOperations();
		RemoveUnusedOperations();
		return no_changes;
	});
#endif

	RunCompilationPass("AddKernelGlobalStoreOperations", [&]() { AddKernelGlobalStoreOperations(); });
	RunCompilationPass("RemoveUnusedKernels", [&]() { RemoveUnusedKernels(); }, true);
	RunCompilationPass("AddMemoryOpIndices", [&]() { AddMemoryOpIndices(); });
	RunCompilationPass("ReorderOperations", [&]() { ReorderOperations(); });
	RunCompilationPass("OptimizeOperations", [&]() { OptimizeOperations(); });
	RunCompilationPass("AddMemoryOpIndices", [&]() { AddMemoryOpIndices(); }, true);

	//TODO: Add support for unrolling small constant kernel dimensions
	//RunCompilationPass("UnrollOperations", [&]() { UnrollOperations(); });
	//RunCompilationPass("SqueezeKernelShapes", [&]() { SqueezeKernelShapes(); });

	RunCompilationPass("FinalizeMemoryIndexing", [&]() { FinalizeMemoryIndexing(); });
	RunCompilationPass("RemoveUnusedOperations", [&]() { RemoveUnusedOperations(); });
	RunCompilationPass("OptimizeKernels", [&]() { OptimizeKernels(); });
	RunCompilationPass("OptimizeHost", [&]() { OptimizeHost(); });
	RunCompilationPass("RemoveUnusedOperations", [&]() { RemoveUnusedOperations(); });

	RunCompilationPass("RemoveUnusedKernels", [&]() { RemoveUnusedKernels(); }, true);
	RunCompilationPass("AddMemoryDeallocation", [&]() { AddMemoryDeallocation(); }, true);
	RunCompilationPass("GetOutputList", [&]() { GetOutputList(); });
	RunCompilationPass("ComputeStatistics", [&]() { ComputeStatistics(); });

#ifdef PROFILE_COMPILATION
	cout << "Profiled compilation passes:" << endl;
	for (const PassStats& stats : pass_stats) {
		cout << "Pass: " << stats.pass_name << " took " << stats.duration << "ms and processed " << stats.node_count << " nodes" << endl;
	}
#endif
}

int GetAxis(int dims, int axis)
{
	if (axis < 0)
	{
		axis = dims + axis;
	}
	return axis;
}

}  // namespace TensorFrost