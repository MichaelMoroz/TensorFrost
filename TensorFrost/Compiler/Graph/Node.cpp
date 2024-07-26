#include "IR.h"

namespace TensorFrost {

int Node::global_index = 0;

void ArgumentManager::UpdateOutputs() {
    for (auto& [id, node] : inputs_) {
        node->args.AddOutput(id, node_);
    }
}

const Tensor *ArgumentManager::GetTensor(ArgType type, int index) const {
    return Get(type, index)->GetTensor();
}

const Tensor & ArgumentManager::operator[](int index) const {
    return *GetTensor(ArgType::Input, index);
}

map<int, const Tensor *> ArgumentManager::GetTensors(ArgType type) const {
    map<int, const Tensor *> tensors;
    for (auto& [id, node] : inputs_) {
        if (id.first == type) {
            tensors[id.second] = node->GetTensor();
        }
    }
    return tensors;
}

ArgumentManager::~ArgumentManager() {

}

bool ArgumentManager::CannotMoveArgument(ArgID id) {
    Node* from = inputs_[id];
    Node* to = node_;
    return (id.first == ArgType::Memory &&
            !to->op->HasAllTypes(OpProp::Set)) ||
           (id.first  == ArgType::Shape && !to->op->HasAllTypes(OpProp::Memory)) ||
           from->op->HasAllTypes(OpProp::Memory) ||
           (from->name == "const" && to->op->HasAllTypes(OpProp::Memory)); //FIX THIS
}

bool ArgumentManager::CannotCopyArgument(ArgID id) {
    Node* from = inputs_[id];
    Node* to = node_;
    bool shape = id.first == ArgType::Shape;
    bool to_memory = to->op->HasAllTypes(OpProp::Memory);
    bool shape_not_memory = shape && !to_memory;
    return id.first == ArgType::Memory || shape_not_memory ||
           from->op->HasAllTypes(OpProp::Static) ||
           from->op->HasAllTypes(OpProp::Memory) || from->flags.has(NodeProp::Modified);
}

bool ArgumentManager::IsChangingInput(ArgID arg) {
    return arg.first == ArgType::Memory &&
           node_->op->HasAllTypes(OpProp::Modifier);
}

void ArgumentManager::UpdateArgument(ArgID id, Node *node) {
    if(node == nullptr) {
        throw std::runtime_error("Node is null");
    }
    if(!Has(id)) {
        throw std::runtime_error("No argument to update");
    }
    inputs_[id] = node;
    argument_types_[id] = node->type;
}

Node* Node::GetLastChild() {
    NodeIterator it = NodeIterator(this);
    for (; !it.end(); it.go_to_next()) {}
    return it.get();
}

const map<NodeProp, string> flag_names = {
    {NodeProp::Modified, "Modified"}, {NodeProp::Placeholder, "Placeholder"},
    {NodeProp::DetachGrad, "DetachGrad"}, {NodeProp::PassGrad, "PassGrad"},
    {NodeProp::KeepDims, "KeepDims"}, {NodeProp::IsStatic, "IsStatic"},
    {NodeProp::OutputMemory, "OutputMemory"}, {NodeProp::InputMemory, "InputMemory"},
    {NodeProp::InputMemoryList, "InputMemoryList"}, {NodeProp::InputShapeMemory, "InputShapeMemory"},
    {NodeProp::InputShapeDim, "InputShapeDim"},
};

string NodeFlagsToString(NodeProp flags) {
    return flag_names.at(flags);
}


} // namespace TensorFrost