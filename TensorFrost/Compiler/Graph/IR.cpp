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
            for (Node* child : to_delete) {
                RemoveNode(child);
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

#define PROFILE_COMPILATION

void IR::RunCompilationPass(string pass_name, const function<void()>& expression, bool print, bool update_graph) {
#ifdef PROFILE_COMPILATION
	auto start = std::chrono::high_resolution_clock::now();
#endif

	expression();

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
}

void IR::CompileIR()
{
	// TODO (Moroz): Add auto tests into build system

	CheckIR("Input", false, false);
	RunCompilationPass("GetInputList", [&]() { GetInputList(); });
	RunCompilationPass("OptimizeOperations", [&]() { OptimizeOperations(); });
	RunCompilationPass("UnrollLoops", [&]() { UnrollLoops(); });
	RunCompilationPass("TryReplaceModificationsWithVersions", [&]() { TryReplaceModificationsWithVersions(); });
	RunCompilationPass("RemoveUnusedOperations", [&]() { RemoveUnusedOperations(); });
	RunCompilationPass("ComputeAutodiff", [&]() { ComputeAutodiff(); });
	RunCompilationPass("RemoveUnusedOperations", [&]() { RemoveUnusedOperations(); }, true);
	RunCompilationPass("UnrollAtomicOperations", [&]() { UnrollAtomicOperations(); });
	RunCompilationPass("InsertAlgorithmicPrimitives", [&]() { InsertAlgorithmicPrimitives(); }, true);
	RunCompilationPass("TryReplaceModificationsWithVersions", [&]() { TryReplaceModificationsWithVersions(); });
	RunCompilationPass("OptimizeOperations", [&]() { OptimizeOperations(); });
	RunCompilationPass("RemoveUnusedOperations", [&]() { RemoveUnusedOperations(); }, true);
	RunCompilationPass("SeparateOperationsIntoKernels", [&]() { SeparateOperationsIntoKernels(); });
	RunCompilationPass("CheckKernelShapes", [&]() { CheckKernelShapes(); }, true);
	RunCompilationPass("ReorderOperations", [&]() { ReorderOperations(); });
	RunCompilationPass("MoveShapeOutsideKernels", [&]() { MoveShapeOutsideKernels(); });
	RunCompilationPass("OptimizeKernels", [&]() { OptimizeKernels(); });
	RunCompilationPass("OptimizeHost", [&]() { OptimizeHost(); });
	RunCompilationPass("UnrollLoops", [&]() { UnrollLoops(1); });
	RunCompilationPass("TryReplaceModificationsWithVersions", [&]() { TryReplaceModificationsWithVersions(); }, true);
	RunCompilationPass("RemoveUnusedOperations", [&]() { RemoveUnusedOperations(); });
	RunCompilationPass("Load fusion", [&]() {
		for (int i = 0; i < 20; i++) {
			RunCompilationPass("AddKernelGlobalLoadOperations", [&]() { AddKernelGlobalLoadOperations(); });
			RunCompilationPass("AddMemoryOpIndices", [&]() { AddMemoryOpIndices(); });
			RunCompilationPass("OptimizeKernelLoadOperations", [&]() { OptimizeKernelLoadOperations(); });
			RunCompilationPass("RemoveUnusedOperations", [&]() { RemoveUnusedOperations(); });
		}
	});
	RunCompilationPass("AddKernelGlobalStoreOperations", [&]() { AddKernelGlobalStoreOperations(); });
	RunCompilationPass("RemoveUnusedKernels", [&]() { RemoveUnusedKernels(); }, true);
	RunCompilationPass("AddMemoryOpIndices", [&]() { AddMemoryOpIndices(); });
	RunCompilationPass("ReorderOperations", [&]() { ReorderOperations(); });
	RunCompilationPass("OptimizeOperations", [&]() { OptimizeOperations(); });
	RunCompilationPass("AddMemoryOpIndices", [&]() { AddMemoryOpIndices(); }, true);
	RunCompilationPass("FinalizeMemoryIndexing", [&]() { FinalizeMemoryIndexing(); });
	RunCompilationPass("RemoveUnusedOperations", [&]() { RemoveUnusedOperations(); });
	RunCompilationPass("OptimizeKernels", [&]() { OptimizeKernels(); });
	RunCompilationPass("OptimizeHost", [&]() { OptimizeHost(); });
	RunCompilationPass("RemoveUnusedOperations", [&]() { RemoveUnusedOperations(); });
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