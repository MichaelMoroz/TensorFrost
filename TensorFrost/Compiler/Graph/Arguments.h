#pragma once

#include "Compiler/Operations.h"
#include "Utility/Utility.h"
#include <set>

namespace TensorFrost {

enum class ArgType {
	Input,
	Index,
	Shape,
	Memory,
	None,
	Count,
};

class Tensor;
class Node;
string TypeToString(ArgType type);

//argument type and index
using ArgID = pair<ArgType, int>;
//input nodes with argument type and index
using NodeArguments = map<ArgID, Node*>;
//argument type and input node
using Arg = pair<ArgID, Node*>;
//argument type and input/output node - edge of the graph
using ArgEdge = pair<Arg, Node*>;

struct HashArgID {
	size_t operator()(const ArgID& id) const {
		return (int)id.first + id.second * (int)ArgType::Count;
	}
};

//set of edges
using ArgEdges = set<ArgEdge>;

#define MAX_ARGS_PER_TYPE 8
#define MAX_ARGS ((int)ArgType::Count * MAX_ARGS_PER_TYPE)

class ArgumentManager {
	Node* node_;
	bool add_parenthesis = false;
	unordered_map<ArgID, TFType, HashArgID> argument_types_;
	unordered_map<ArgType, int> argument_counts_;
	unordered_map<ArgID, string, HashArgID> argument_names_;
	unordered_map<ArgID, bool, HashArgID> argument_requires_parenthesis_;
	unordered_map<ArgID, Node*, HashArgID> inputs_;
	ArgEdges outputs_;

	void AddOutput(ArgID id, Node* node) {
		outputs_.insert({{id, node_}, node});
	}

	void RemoveOutput(ArgID id, Node* node) {
		if (!outputs_.contains({{id, node_}, node})) {
			throw std::runtime_error("Output does not exist");
		}
		outputs_.erase({{id, node_}, node});
	}

	void UpdateOutputs();
	void ClearOutputs();
public:
	ArgumentManager(Node* node) {
		if (node == nullptr) {
			throw std::runtime_error("Node is null");
		}
		this->node_ = node;
	}

	void AddParenthesis(bool add) {
		add_parenthesis = add;
	}

	unordered_map<ArgID, Node*, HashArgID> Inputs() const {
		return inputs_;
	}

	ArgEdges Outputs() const {
		return outputs_;
	}

	void AddArgument(ArgID id, Node *node);
	void AddArgument(ArgType type, int index, Node *node) {
		AddArgument(ArgID(type, index), node);
	}

	void UpdateArgument(ArgID id, Node *node);

	void AddArguments(NodeArguments new_args) {
		for (auto& [id, node] : new_args) {
			AddArgument(id, node);
		}
	}

	void SetName(ArgID id, string name, bool requires_parenthesis = false) {
		argument_names_[id] = name;
	    argument_requires_parenthesis_[id] = requires_parenthesis;
	}

	bool Has(ArgID id) const {
		return inputs_.find(id) != inputs_.end();
	}

	bool Has(ArgType type, int index = 0) const {
		ArgID id = ArgID(type, index);
		return Has(id);
	}

	Node* Get(ArgType type, int index = 0) const {
		ArgID id = ArgID(type, index);
		auto Arg = inputs_.find(id);
		if (Arg != inputs_.end()) {
			return Arg->second;
		} else {
			throw std::runtime_error("Argument of type " + TypeToString(type) + " at index " + std::to_string(index) + " not found");
		}
	}

	void Remove(ArgID id);
	void Remove(ArgType type, int index = 0) {
		Remove(ArgID(type, index));
	}

	const Tensor *GetTensor(ArgType type, int index = 0) const;

	const Tensor& operator[](int index) const;

	TFType Type(ArgType type, int index = 0) const {
		ArgID id = ArgID(type, index);
		auto Arg = argument_types_.find(id);
		if (Arg != argument_types_.end()) {
			return Arg->second;
		}
		else {
			throw std::runtime_error("Argument type not found");
		}
	}

	int Count(ArgType type) const {
		auto Arg = argument_counts_.find(type);
		if (Arg != argument_counts_.end()) {
			return Arg->second;
		}
		else {
			return 0;
		}
	}

	bool RequiresParenthesis(ArgID id) const {
		auto Arg = argument_requires_parenthesis_.find(id);
		if (Arg != argument_requires_parenthesis_.end()) {
			return Arg->second;
		}
		else {
			return false;
		}
	}

	string Name(ArgType type, int index = 0) const {
		ArgID id = ArgID(type, index);
		auto Arg = argument_names_.find(id);
		if (Arg != argument_names_.end()) {
			string name = Arg->second;
			if (add_parenthesis && RequiresParenthesis(id)) {
				name = "(" + name + ")";
			}
			return name;
		}
		else {
			throw std::runtime_error("Argument name not found");
		}
	}

	NodeArguments GetArguments() const {
		NodeArguments arguments;
		for (auto& [id, node] : inputs_) {
			arguments[id] = node;
		}
		return arguments;
	}

	NodeArguments GetArguments(ArgType type) const {
		NodeArguments arguments;
		for (auto& [id, node] : inputs_) {
			if (id.first == type) {
				arguments[id] = node;
			}
		}
		return arguments;
	}

	map<int, const Tensor *> GetTensors(ArgType type) const;

	vector<const Tensor*> GetTensorVector(ArgType type) const;

	~ArgumentManager();

	bool CannotMoveArgument(ArgID id);
	bool CannotCopyArgument(ArgID id);
	bool IsChangingInput(ArgID arg);

	void RemoveArguments(ArgType arg);
};

} // namespace TensorFrost