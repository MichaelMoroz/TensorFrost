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
#include <chrono>

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
		root->index_ = 0;
		root->name = "root";
        root->initialize(nullptr, {}, "host", TFType::None, existing_nodes, true);
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
		Node* newNode = nullptr;
        if (cursor->valid()) { //already initialized, add new node before cursor
            newNode = new Node(cursor->prev, cursor->parent);
			if (cursor->prev) 
				cursor->prev->next = newNode;
			else if (cursor->parent) 
				cursor->parent->child = newNode;
            cursor->prev = newNode;
			newNode->next = *cursor;
            newNode->initialize(tensor, std::move(args), std::move(name), type, existing_nodes);
        } else {
        	newNode = cursor.get();
            cursor->initialize(tensor, std::move(args), std::move(name), type, existing_nodes);
			cursor.go_to_next();
        }

#ifndef NDEBUG
		newNode->created_in_pass = current_pass;
		newNode->created_in_function = current_function;
#endif

		return newNode;
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
    	if(node != nullptr) {
    		cursor = NodeIterator(node, root);
    	} else {
    		throw std::runtime_error("Cursor cannot be set to null");
		}
    }

	bool LimitKernelMemoryDependencies();

	void UnrollOperations();

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

	void CheckIR(string name, bool check_clustering, bool check_kernels);
	string PrintListing(map<Node*, string> node_debug) const;

	string GetNodeListing(Node *node) const;

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
	                          unordered_map<int, Node*> indices, Node* cursor = nullptr);
	void ReorderOperations();
	void MoveShapeOutsideKernels();
	bool OptimizeKernels();
	void OptimizeHost();
	void OptimizeOperations();
	void OptimizeHostValuesWithHints();

	bool OptimizeKernelLoadOperations();
	void OptimizeReductions();

	unordered_set<Node *> GetDependencies(unordered_set<Node *> nodes);

	void RemoveUnusedOperations();

	bool InsertAlgorithmicPrimitives(bool skip_differentiable);
	void UnrollLoops(int max_iterations = 8);
	void UnrollAtomicOperations();
	void TryReplaceModificationsWithVersions();
	bool ComputeAutodiff();
	void SeparateOperationsIntoKernels();
	void ComputeNodeCost();

	map<Node *, ArgEdges> GetKernelOutputs(Node *kernel);
	void AddNodeLoadOperations(Node* node, Node* kernel, Tensors indices);
	void AddKernelGlobalLoadOperations();
	void AddMemoryOpIndices();
	void AddKernelGlobalStoreOperations();

	unordered_set<Node*> ComputeKernelDependencies(Node* kernel);

	void CheckKernelShapes();
	void UpdateKernelShapes();
	void AddMemoryDeallocation();
	void RunCompilationPass(string pass_name, const function<void()> &expression, bool print = false, bool update_graph = false);

	bool RunIterativeCompilationPass(string pass_name, int max_iterations, const function<bool()> &expression,
	                                 bool print = false,
	                                 bool update_graph = false);

	void ReplaceDimNodes(Node* kernel, vector<Tensor*> indices, int dims);
	void MultiDimensionalModeIndices(vector<Tensor*>& indices, Node* kernel_,
	                                 int dims, Tensors kernel_shape);
	Tensor* LinearBlockModeIndices(vector<Tensor*>& indices, Node* kernel_, int dims,
	                            Tensors kernel_shape);

	void ComputeAddress(Node *node, vector<Tensor *> indices);

	void FinalizeMemoryIndexing();
	void RemoveUnusedKernels();
	void CompileIR();

	void UpdateIndex() {
		int index = 0;
		for (auto node = begin(); !node.end(); node.next()) {
			node->UpdateEdges();
			node->index_ = index++;
		}

		index++; //add root node

		if (index != existing_nodes.size()) {
			unordered_set<Node*> found_nodes;
			found_nodes.insert(root);
			for (auto node = begin(); !node.end(); node.next()) {
				found_nodes.insert(*node);
			}

			unordered_set<Node*> missing_nodes;
			for (auto node : existing_nodes) {
				if (found_nodes.find(node) == found_nodes.end()) {
					missing_nodes.insert(node);
				}
			}

			string missing_nodes_str = "";
			for (auto node : missing_nodes) {
				missing_nodes_str += GetNodeListing(node) + "\n";
			}

			missing_nodes_str += "\n" + PrintListing({});

			throw std::runtime_error("\n Some nodes got lost during indexing. Expected " + to_string(existing_nodes.size()) + " but got " + to_string(index) + ". Likely invalid graph. Missing nodes:\n" + missing_nodes_str);
		}
	}

	void UpdateGraph(const Node* uroot = nullptr) {
		if (uroot == nullptr) {
			uroot = root;
		}

		UpdateIndex();

#ifdef _DEBUG
		map<Node*, string> invalid_nodes;
		// check if graph is valid
		for (auto node = NodeIterator(uroot); !node.end(); node.next()) {
			// if there are null inputs throw an error
			for (auto& [id, n] : (*node)->args.Inputs()) {
				if (n == nullptr) {
					throw std::runtime_error("Null input found in node " + (*node)->var_name + ". Likely an icorrectly deleted node.");
				} else if (n->index_ > (*node)->index_) { //if input node is after current node, throw an error
					invalid_nodes[*node] = "Argument " + TypeToString(id.first) + ":" +
										to_string(id.second) + " " + n->var_name + " is after current node";
				}
			}
		}

		if(invalid_nodes.size() > 0) {
#ifdef NDEBUG
			std::string error = "Invalid graph: ";
			for (auto [node, message] : invalid_nodes) {
				error += GetNodeListing(node) + ": " + message + "\n";
			}
			throw std::runtime_error(error);
#else
			throw std::runtime_error("Invalid graph: " + PrintListing(invalid_nodes));
#endif
		}

#endif

		//update modified flags
		for (auto node = NodeIterator(uroot); !node.end(); node.next()) {
			node->flags.remove(NodeProp::Modified);
			//go over all outputs and check if they are modifiers
			for (auto [edge, to] : node->args.Outputs()) {
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

	template <typename... Args>
	vector<Node*> GetNodesOfType(OpProp type, Args... args) const {
		vector<Node*> result;
		for (auto node = begin(); !node.end(); node.next()) {
			if (node->op->HasAllTypes(type, args...)) {
				result.push_back(*node);
			}
		}
		return result;
	}

	size_t CountNodesOfType(OpProp type) const {
		size_t count = 0;
		for (auto node = begin(); !node.end(); node.next()) {
			if (node->op->HasAllTypes(type)) {
				count++;
			}
		}
		return count;
	}

	vector<Node*> GetChildren(Node* node) const {
		vector<Node*> result;
		for (auto child = NodeIterator(node); !child.end(); child.next()) {
			result.push_back(*child);
		}
		return result;
	}

	size_t CombineHashes(size_t hash1, size_t hash2) const {
		return hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
	}

	size_t GetApproximateStateHash() const {
		size_t hash = 0;
		for (auto node = begin(); !node.end(); node.next()) {
			size_t node_hash1 = std::hash<std::string>{}(node->name);
			size_t node_hash2 = std::hash<std::string>{}(node->debug_name);
			size_t node_hash3 = std::hash<int>{}(node->debug_index);
			hash = CombineHashes(hash, CombineHashes(node_hash1, CombineHashes(node_hash2, node_hash3)));
		}
		return hash;
	}

	int input_memory_count = 0;
	int output_memory_count = 0;
	int temp_memory_count = 0;

	int readbacks = 0;
	int writebacks = 0;

	unordered_map<Node*, unordered_map<int, Node*>> shape_memory_map;
	unordered_map<int, Node*> input_memory_map;
	unordered_map<int, Node*> output_memory_map;

	string current_pass = "Tracing initial graph";
	string current_function = "None";

	struct PassStats {
		string pass_name;
		float duration;
		int node_count;
	};
	vector<PassStats> pass_stats;

	void ReplaceArgs(const ArgEdges& edges, const map<Node*, Node*>& replacements) {
		edgesToUpdate.insert(edges.begin(), edges.end());
		replacementNodes.insert(replacements.begin(), replacements.end());
	}

	void RemoveNodes(const vector<Node*>& nodes) {
		removedNodes.insert(removedNodes.end(), nodes.begin(), nodes.end());
	}

	void ApplyChanges(bool update_graph = true, const Node *uroot = nullptr);
	void ClearChanges();

	ArgEdges edgesToUpdate{};
	map<Node*, Node*> replacementNodes{};
	vector<Node*> removedNodes{};
	unordered_set<Node*> existing_nodes{};

	static int max_kernel_memory_dependencies;
	static int max_allowed_memory_dependencies;
};

int GetAxis(int dims, int axis);

}  // namespace TensorFrost