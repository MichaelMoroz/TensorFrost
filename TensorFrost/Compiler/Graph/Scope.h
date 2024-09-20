#pragma once

#include "Compiler/Operations.h"
#include "Utility/Utility.h"
#include "Node.h"

namespace TensorFrost {

using Tensors = vector<const Tensor*>;

class ShapeInfo {
 public:
	vector<pair<Node*, bool>> shape;
	int dim = 0;
	string name;

	ShapeInfo() {}

	ShapeInfo(ShapeInfo* shape_info) {
		shape = shape_info->shape;
		dim = shape_info->dim;
		name = shape_info->name;
	}

	ShapeInfo(const Node* node) {
		dim = node->args.Count(ArgType::Shape);
		for (int i = 0; i < dim; i++) {
			AddShape(i, node->args.Get(ArgType::Shape, i));
		}
		this->name = node->var_name != "" ? node->var_name : node->name;
	}

	Node* operator[](int index) const {
		return shape[index].first;
	}

	void AddShape(int index, Node* node) {
		if(shape.size() <= index) {
			shape.resize(index + 1);
		}
		shape[index] = {node, false};
		dim = max(dim, index + 1);
	}

	bool CheckValidity(bool throw_error = false) const {
		for (auto node : shape) {
			if(node.first == nullptr) {
				if (throw_error) {
					throw std::runtime_error("Shape not fully defined");
				}
				return false;
			}
		}
		return true;
	}

	const Tensor* GetTensor(int index) const {
		CheckValidity(true);
		return shape[index].first->GetTensor();
	}

	bool IsExpanded(int index) const {
		return shape[index].second;
	}

	Tensors GetTensors() const {
		CheckValidity(true);
		Tensors tensors = Tensors();
		for (auto node : shape) {
			tensors.push_back(node.first->GetTensor());
		}
		return tensors;
	}

	NodeArguments GetArguments() const {
		CheckValidity(true);
		NodeArguments arguments;
		for (int i = 0; i < shape.size(); i++) {
			arguments[ArgID(ArgType::Shape, i)] = shape[i].first;
		}
		return arguments;
	}

	vector<int> GetShape(int default_value = 256) const;

	static float GetSizeEstimate(ShapeInfo &shape);

	void InsertDim(int index, Node* node, bool expanded = false) {
		if (index >= shape.size()+1) {
			shape.resize(index + 1);
		}
		shape.insert(shape.begin() + index, {node, expanded});
		dim++;
	}

	void ExpandDimensionsTo(int new_dim);
};

struct ShapeCompareResult {
	bool compatible;
	ShapeInfo broadcast_shape;
	bool broadcast;
	int broadcast_dim;
	int a_dim;
	int b_dim;
	int min_dim;
};

struct ShapeDimCompareResult {
	bool compatible;
	Node* broadcast_dim;
	bool broadcast;
	int a_dim;
	int b_dim;
};

ShapeCompareResult CompareShape(const Node* a, const Node* b,
                                bool exact_match = false,
                                bool throw_error = false);

ShapeCompareResult CompareShape(ShapeInfo& a, ShapeInfo& b,
                                bool exact_match = false,
                                bool throw_error = false);

ShapeDimCompareResult CompareShapeDim(Node* a_node, Node* b_node, bool exact_match = false);

/// <summary>
/// Class to select kernel scopes from the IR graph given the constraints and the root node
/// </summary>
class KernelScope {
 public:
	Node* begin = nullptr;
	Node* end = nullptr;
	ShapeInfo scope_shape;
	unordered_set<Node*> boundary_nodes;

	static bool IsBoundary(const Node* input, const Node* output, ArgType arg_type, bool is_identity);

	KernelScope() : begin(nullptr), end(nullptr) {}
	KernelScope(Node* node, unordered_set<KernelScope*>& output_scopes);

	KernelScope(Node* begin, Node* end, ShapeInfo shape, unordered_set<Node*> boundary_nodes)
	    : begin(begin), end(end), scope_shape(shape), boundary_nodes(boundary_nodes) {}

	void CopyProperties(KernelScope* other) {
		begin = other->begin;
		end = other->end;
		scope_shape = other->scope_shape;
		boundary_nodes = other->boundary_nodes;
	}

	bool IsValid() const;

	static pair<std::unordered_set<KernelScope *>, bool> ComputeScopes(Node *root);

	static KernelScope* Merge(KernelScope* a, KernelScope* b);

	void CreateKernel();

	void AddBoundaryNodes(unordered_set<Node*> new_boundary_nodes) {
		boundary_nodes.insert(new_boundary_nodes.begin(), new_boundary_nodes.end());
	}
};

}  // namespace TensorFrost
