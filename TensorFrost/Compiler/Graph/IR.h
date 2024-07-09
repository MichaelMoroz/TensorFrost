#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <stack>
#include <set>
#include <bitset>
#include <array>

#include "Compiler/Operations.h"
#include "Utility/Utility.h"
#include "Node.h"
#include "Scope.h"

namespace TensorFrost {

class IR {
public:
	Node* root;
    NodeIterator cursor;

	IR() {
        root = new Node();
        root->initialize(nullptr, {}, "host", TFType::None, true);
        cursor = NodeIterator(root);
    }

    ~IR() {
        vector<Node*> to_delete;
		for (auto node = begin(); !node.end(); node.next()) {
			to_delete.push_back(*node);
        }
        for (Node* node : to_delete) {
            delete node;
        }
		delete root;
    }

    NodeIterator begin() const {
        return NodeIterator(root);
    }

    Node* AddNode(Tensor* tensor, NodeArguments&& args, string&& name, TFType type) {
        if (cursor->valid()) { //already initialized, add new node before cursor
            Node* newNode = new Node(cursor->prev, cursor->parent);
			if (cursor->prev) 
				cursor->prev->next = newNode;
			else if (cursor->parent) 
				cursor->parent->child = newNode;
            cursor->prev = newNode;
			newNode->next = *cursor;
            newNode->initialize(tensor, std::move(args), std::move(name), type);
			return newNode;
        } else {
            cursor->initialize(tensor, std::move(args), std::move(name), type);
			cursor.go_to_next();
			return cursor->prev;
        }
    }

	void MoveNodeTo(Node* target_place, Node* note_to_move) {
		if (note_to_move->valid()) {
			//remove from current position
			if (note_to_move->parent && note_to_move->parent->child == note_to_move) {
				note_to_move->parent->child = note_to_move->next;
			}
			else if (note_to_move->prev) {
				note_to_move->prev->next = note_to_move->next;
			}
			note_to_move->next->prev = note_to_move->prev;

			//insert into new position
			note_to_move->parent = target_place->parent;
			note_to_move->prev = target_place->prev;
			note_to_move->next = target_place;
			if (target_place->prev) {
				target_place->prev->next = note_to_move;
			}
			else if (target_place->parent) {
				target_place->parent->child = note_to_move;
			}
			target_place->prev = note_to_move;
		}
	}

    void RemoveNode(Node* node);

    void SetCursor(Node* node) {
        cursor = NodeIterator(node, root);
    }

	stack<Node*> scope_stack;

	void EndScope() {
		if (scope_stack.empty()) throw std::runtime_error("No scope to end");
		SetCursor(scope_stack.top());
		scope_stack.pop();
	}

	void BeginScope(Node* node) {
		scope_stack.push(*cursor);
		SetCursor(node);
	}

	void BeginScopeLastChild(Node* node) {
		BeginScope(node->GetLastChild());
	}

    void ExecuteExpressionAfter(Node* node, const function<void()>&& expression) {
		BeginScope(node->next);
		expression();
		EndScope();
    }

    void ExecuteExpressionBefore(Node* node, const function<void()>&& expression) {
		BeginScope(node);
		expression();
		EndScope();
    }

	void ExecuteExpressionFirstChild(Node* node, const function<void()>&& expression) {
		BeginScope(node->child);
		expression();
		EndScope();
	}

	void ExecuteExpressionLastChild(Node* node, const function<void()>&& expression) {
		BeginScopeLastChild(node);
		expression();
		EndScope();
	}

	void CheckIR(string name, bool check_clustering, bool check_kernels) const;
	string PrintListing(map<Node*, string> node_debug) const;
	map<Node*, Node*> CopyNodes(set<Node*> nodes_to_copy,
	                            unordered_map<Node*, Node*> argument_replacements,
	                            unordered_map<int, Node*> indices,
	                            unordered_set<Node*> targets, bool must_copy_all);
	map<Node*, Node*> CopyComputation(const unordered_set<Node*>& targets,
	                                  const unordered_map<int, Node*>& indices);
	void GetInputList();
	void GetOutputList();
	void ComputeStatistics();
	void CopyArguments(ArgEdges args_to_copy, Node *cursor);
	map<Node*, Node*> CopyNodesWithIndex(unordered_set<Node*> nodes_to_copy,
	                          unordered_map<int, Node*> indices, Node* cursor);
	void ReorderOperations();
	void MoveShapeOutsideKernels();
	void OptimizeKernels();
	void OptimizeHost();
	void OptimizeOperations();
	void OptimizeKernelLoadOperations();

	unordered_set<Node *> GetDependencies(unordered_set<Node *> nodes);

	void RemoveUnusedOperations();
	void InsertAlgorithmicPrimitives();
	void UnrollLoops();
	void TryReplaceModificationsWithVersions();
	void ComputeAutodiff();
	void SeparateOperationsIntoKernels();
	void ComputeNodeCost();

	map<Node *, ArgEdges> GetKernelOutputs(Node *kernel);
	void AddNodeLoadOperations(Node* node, Node* kernel, Tensors indices);
	void AddKernelGlobalLoadOperations();
	void AddMemoryOpIndices();
	void AddKernelGlobalStoreOperations();
	void CheckKernelShapes();
	void AddMemoryDeallocation();
	void ReplaceDimNodes(Node* kernel, vector<Tensor*> indices, int dims);
	void MultiDimensionalModeIndices(vector<Tensor*>& indices, Node* kernel_,
	                                 int dims, Tensors kernel_shape);
	Tensor* LinearBlockModeIndices(vector<Tensor*>& indices, Node* kernel_, int dims,
	                            Tensors kernel_shape);
	void FinalizeMemoryIndexing();
	void RemoveUnusedKernels();
	void CompileIR();

	void UpdateGraph() const {
		// update edges
		for (auto node = begin(); !node.end(); node.next()) {
			node->UpdateEdges();
			node->args.ClearOutputs();
		}

		int index = 0;
		for (auto node = begin(); !node.end(); node.next()) {
			node->index_ = index++;
		}

		map<Node*, string> invalid_nodes;
		// check if graph is valid
		for (auto node = begin(); !node.end(); node.next()) {
			// if there are null inputs throw an error
			for (auto& [id, n] : (*node)->args.inputs_) {
				if (n == nullptr) {
					throw std::runtime_error("Null input found in node " + (*node)->var_name + ". Likely an icorrectly deleted node.");
				} else if (n->index_ > (*node)->index_) { //if input node is after current node, throw an error
					invalid_nodes[*node] = "Argument " + TypeToString(id.first) + ":" +
										to_string(id.second) + " " + n->var_name + " is after current node";
				}
			}
		}

		if(invalid_nodes.size() > 0) {
			throw std::runtime_error("Invalid graph: " + PrintListing(invalid_nodes));
		}

		// update outputs
		for (auto node = begin(); !node.end(); node.next()) {
			node->args.UpdateOutputs();
		}

		//update modified flags
		for (auto node = begin(); !node.end(); node.next()) {
			node->flags.remove(NodeProp::Modified);
			//go over all outputs and check if they are modifiers
			for (auto [edge, to] : node->args.outputs_) {
				auto& [id, from] = edge;
				if (to->op->HasAllTypes(OpProp::Modifier)) {
					bool is_memory = false;
					if (id.first != ArgType::Memory) {
						is_memory = true;
					}
					if (!is_memory) {
						node->flags.set(NodeProp::Modified);
						break;
					}
				}
			}
		}
	}

	vector<Node*> GetNodesOfType(const string& name) const {
		vector<Node*> result;
		for (auto node = begin(); !node.end(); node.next()) {
			if (node->name == name) {
				result.push_back(*node);
			}
		}
		return result;
	}

	vector<Node*> GetNodesOfType(OpProp type) const {
		vector<Node*> result;
		for (auto node = begin(); !node.end(); node.next()) {
			if (node->op->HasAllTypes(type)) {
				result.push_back(*node);
			}
		}
		return result;
	}

	vector<Node*> GetChildren(Node* node) const {
		vector<Node*> result;
		for (auto child = NodeIterator(node); !child.end(); child.next()) {
			result.push_back(*child);
		}
		return result;
	}

	int input_memory_count = 0;
	int output_memory_count = 0;
	int temp_memory_count = 0;

	int readbacks = 0;
	int writebacks = 0;

	unordered_map<Node*, unordered_map<int, Node*>> shape_memory_map;
	unordered_map<int, Node*> input_memory_map;
	unordered_map<int, Node*> output_memory_map;
};

int GetAxis(int dims, int axis);

}  // namespace TensorFrost