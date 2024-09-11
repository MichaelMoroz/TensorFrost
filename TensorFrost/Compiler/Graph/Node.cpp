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
    bool is_output = from->flags.has(NodeProp::OutputMemory);
    bool is_input = from->flags.has(NodeProp::InputMemory);
    bool no_fusion = from->flags.has(NodeProp::StopFusion);
    return id.first == ArgType::Memory || shape_not_memory ||
           from->op->HasAllTypes(OpProp::Static) || from->op->HasAllTypes(OpProp::Memory) ||
           from->flags.has(NodeProp::Modified) || is_output || is_input || no_fusion;
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

Node * Node::GetChild(string name) {
    for(NodeIterator it = NodeIterator(this); !it.end(); it.next()) {
        if(it->name == name) {
            return it.get();
        }
    }
    return nullptr;
}

Node * Node::GetNodeWithCommonParent(Node *other) {
    for (Node* cur_parent = this; cur_parent != nullptr; cur_parent = cur_parent->parent) {
        if (cur_parent->parent == other->parent) {
            return cur_parent;
        }
    }
    for (Node* cur_parent = other; cur_parent != nullptr; cur_parent = cur_parent->parent) {
        if (cur_parent->parent == this->parent) {
            return cur_parent;
        }
    }
    for (Node* cur_parent1 = this; cur_parent1 != nullptr;
         cur_parent1 = cur_parent1->parent) {
        for (Node* cur_parent2 = other; cur_parent2 != nullptr;
             cur_parent2 = cur_parent2->parent) {
            if (cur_parent1->parent == cur_parent2->parent) {
                return cur_parent1;
            }
        }
    }
    throw std::runtime_error("No common parent found");
}

Node* Node::GetLastChild() {
    NodeIterator it = NodeIterator(this);
    for (; !it.end(); it.go_to_next()) {}
    return it.get();
}

bool Node::HasCommonParents(Node *other, int max_depth) const {
    int depth = 0;
    for (Node* cur_parent = parent; cur_parent != nullptr; cur_parent = cur_parent->parent) {
        if (depth++ > max_depth) {
            break;
        }
        if (!other->HasParent(cur_parent)) {
            return false;
        }
    }
    return true;
}

bool Node::HasParent(string name) {
    return GetParent(name) != this;
}

bool Node::HasChild(string name) {
    return GetChild(name) != nullptr;
}

void Node::ValidateParentShapes() const {
    //compare the shape of this node with the shape of all its parents
    ShapeInfo shape = ShapeInfo(this);
    for (Node* cur_parent = parent; cur_parent != nullptr; cur_parent = cur_parent->parent) {
        ShapeInfo parent_shape = ShapeInfo(cur_parent);
        ShapeCompareResult result = CompareShape(parent_shape, shape); //must only be broadcastable
        if (!result.compatible) {
            throw std::runtime_error(MakeNodeErrorMessage("The node " + debug_name + " (" + name + ") has incompatible shapes with its parent " +
                cur_parent->debug_name + " (" + cur_parent->name + ")", {this, cur_parent}));
        }
    }
}

void Node::SetMemoryType(NodeProp memory_type, int index) {
    flags.set(memory_type, index);
}

void Node::CheckNode() const {
    // must have operation
    if (op == nullptr) {
        throw std::runtime_error("Operation object not found");
    }

    // must have tensor
    if (tensor_ == nullptr && !flags.has(NodeProp::IsStatic)) {
        throw std::runtime_error("Tensor not found");
    }

    //validate the shape of the node if its not scalar
    if (args.Count(ArgType::Shape) > 0) {
        ValidateParentShapes();
    }
}

Node * Node::GetLastVersion(Node *latest_node) {
    //find last store/scatter operation
    Node* last_modifier = this;
    int last_index = -1;
    Node* loop_node = latest_node->GetParent("loop");
    bool has_loop = loop_node != latest_node;
    for (auto [edge, to] : args.outputs_) {
        auto& [id, from] = edge;
        bool is_memory = false;
        if (id.first != ArgType::Memory) {
            is_memory = true;
        }
        if (is_memory) {
            continue;
        }
        if (to->op->HasAllTypes(OpProp::Modifier)) {
            if (to->index_>last_index) {
                // either find the last modifier or the last memory node
                // or if there is a loop, find the last modifier inside the loop (i.e.
                // the previous iteration's modifier)
                // if the loop is scalar, then it doesn't matter
                bool before_latest = to->index_ < latest_node->index_;
                bool inside_loop = has_loop && to->HasParent(loop_node);
                bool not_same = to != latest_node;
                if ((before_latest || inside_loop) && not_same)
                {
                    last_index = to->index_;
                    last_modifier = to;
                }
            }
        }
    }
    return last_modifier;
}

Node * Node::GetFinalVersion() {
    Node* final_version = this;
    int last_index = -1;
    for (auto [edge, to] : args.outputs_) {
        auto& [id, from] = edge;
        bool is_memory = false;
        if (id.first != ArgType::Memory) {
            is_memory = true;
        }
        if (is_memory) {
            continue;
        }
        if (to->op->HasAllTypes(OpProp::Modifier) && !to->op->HasAllTypes(OpProp::MemoryOp)) {
            if (to->index_ > last_index) {
                last_index = to->index_;
                final_version = to;
            }
        }
    }
    return final_version;
}

const map<NodeProp, string> flag_names = {
    {NodeProp::Modified, "Modified"}, {NodeProp::Placeholder, "Placeholder"},
    {NodeProp::DetachGrad, "DetachGrad"}, {NodeProp::PassGrad, "PassGrad"},
    {NodeProp::KeepDims, "KeepDims"}, {NodeProp::IsStatic, "IsStatic"},
    {NodeProp::OutputMemory, "OutputMemory"}, {NodeProp::InputMemory, "InputMemory"},
    {NodeProp::InputMemoryList, "InputMemoryList"}, {NodeProp::InputShapeMemory, "InputShapeMemory"},
    {NodeProp::InputShapeDim, "InputShapeDim"}, {NodeProp::NoLoadFusion, "NoLoadFusion"}, {NodeProp::StopFusion, "StopFusion"}
};

string NodeFlagsToString(NodeProp flags) {
    if (!flag_names.contains(flags)) {
        throw std::runtime_error("Flag name not defined");
    }
    return flag_names.at(flags);
}

void Node::UpdateEdges() {
    if (!child) child = new Node(nullptr, this);
    if (!next) next = new Node(this, parent);
    if (child->valid()) {
        child->parent = this;
    }
    if (next->valid()) {
        next->prev = this;
        next->parent = parent;
    }
}

const map<IndexingMode, string> indexing_mode_names = {
    {IndexingMode::Clamp, "Clamp"}, {IndexingMode::Repeat, "Repeat"},
    {IndexingMode::Mirror, "Mirror"}, {IndexingMode::Unsafe, "Unsafe"}
};

string IndexingModeToString(IndexingMode mode) {
    return indexing_mode_names.at(mode);
}

void Node::initialize(Tensor *tensor, NodeArguments &&new_args, string &&new_name, TFType new_type, bool set_static) {
    if(valid()) {
        throw runtime_error("Node already initialized");
    }
    UpdateEdges();
    flags.remove(NodeProp::Placeholder);

    tensor_ = tensor;
    type = new_type;
    args.AddArguments(std::move(new_args));
    args.UpdateOutputs();
    flags.set(NodeProp::IsStatic, set_static);
    name = std::move(new_name);
    op = FindOperation(name);
    CheckNode();
}

void Node::CopyProperties(Node *other) {
    name = other->name;
    debug_name = other->debug_name;
    indexing_mode_ = other->indexing_mode_;
    group_size = other->group_size;
    type = other->type;

    flags.copy_all(other->flags);
}

void Node::CopyMetadata(Node *other) {
    if (other->debug_name != "") {
        debug_name = other->debug_name;
    }
    if(other->indexing_mode_ != IndexingMode::Clamp) {
        indexing_mode_ = other->indexing_mode_;
    }
    group_size = other->group_size;

    flags.copy_all_except(other->flags, {NodeProp::Modified});
}

int Node::ComputeDepth(Node *root) const {
    int depth = 0;
    for (const Node* node = this; node != root; node = node->parent) {
        depth++;
    }
    return depth;
}

bool Node::HasParent(Node *node) const {
    for (Node* cur_parent = parent; cur_parent != nullptr; cur_parent = cur_parent->parent) {
        if (cur_parent == node) {
            return true;
        }
    }
    return false;
}

void Node::ReplaceThisWithGivenNode(Node *replacement, int min_index, bool make_modified, bool copy_metadata) {
    ArgEdges remaining_outputs;
    for (auto [edge, to] : args.outputs_) {
        auto& [id, from] = edge;
        if (to->index_ >= min_index) {
            if(make_modified) {
                replacement->flags.set(NodeProp::Modified);
            }
            to->args.UpdateArgument(id, replacement);
            replacement->args.AddOutput(id, to);
        } else {
            remaining_outputs.push_back({edge, to});
        }
    }

    this->args.outputs_ = remaining_outputs;

    if(copy_metadata) {
        replacement->CopyMetadata(this);
        this->flags.clear();
    }
}

Node * Node::GetParent(string name) {
    for (Node* cur_parent = parent; cur_parent != nullptr; cur_parent = cur_parent->parent) {
        if (cur_parent->name == name) {
            return cur_parent;
        }
    }
    return this;
}
} // namespace TensorFrost