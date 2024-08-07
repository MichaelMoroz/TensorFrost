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

void IR::CompileIR()
{
	// TODO (Moroz): Add auto tests into build system

	CheckIR("Input", false, false);
	GetInputList();
	OptimizeOperations();
	//CheckIR("Optimize operations", false, false);
	UnrollLoops();
	TryReplaceModificationsWithVersions();
	RemoveUnusedOperations();
	//CheckIR("Remove Unused Operations 0", false, false);
	ComputeAutodiff();
	RemoveUnusedOperations();
	CheckIR("Compute Autodiff", false, false);
	UnrollAtomicOperations();
	InsertAlgorithmicPrimitives();
	CheckIR("Insert Algorithmic Primitives", false, false);
	TryReplaceModificationsWithVersions();
	OptimizeOperations();
	RemoveUnusedOperations();
	CheckIR("Remove Unused Operations 1", false, false);
	SeparateOperationsIntoKernels();
	CheckKernelShapes();
	//UnrollDimensions();
	CheckIR("Separate Operations Into Kernels", false, false);
	ReorderOperations();
	CheckIR("Reorder Operations", true, false);
	MoveShapeOutsideKernels();
	OptimizeKernels(); //fuse kernels by copying inputs
	OptimizeHost();
	UnrollLoops(1);
	TryReplaceModificationsWithVersions();
	//UnrollKernelDimensions();
	CheckIR("Optimize kernels and host", true, false);
	RemoveUnusedOperations();
	for (int i = 0; i < 20; i++) { //fusing kernels by loads (tensor product)
		RemoveUnusedOperations();
		AddKernelGlobalLoadOperations();
		AddMemoryOpIndices();
		CheckIR("Load optimization 1 iteration " + to_string(i), true, false);
		OptimizeKernelLoadOperations();
		//CheckIR("Load optimization 2 iteration " + to_string(i), true, false);
	}
	CheckIR("Optimize kernel loads", true, false);
	AddKernelGlobalStoreOperations();
	RemoveUnusedKernels();
	CheckIR("Add Kernel Global Memory Operations", true, true);
	AddMemoryOpIndices();
	ReorderOperations();
	OptimizeOperations();
	AddMemoryOpIndices();
	CheckIR("Final optimization", true, true);
	FinalizeMemoryIndexing();
	RemoveUnusedOperations();
	//CheckIR("Finalize Memory Indexing", false, false);
	OptimizeKernels();
	OptimizeHost();
	//OptimizeLoops();
	RemoveUnusedOperations();
	//CheckIR("Finalize Memory Indexing 2", true, true);
	RemoveUnusedKernels();
	OptimizeOperations();
	RemoveUnusedOperations();
	//CheckIR("Remove Unused Operations 2", true, true);
	AddMemoryDeallocation();
	CheckIR("Add deallocation", true, true);
	GetOutputList();
	ComputeStatistics();
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