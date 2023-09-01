#include <algorithm>

template <typename T1, typename T2, typename T3>
Node<T1, T2, T3>::Node(T1 type, T2 property)
    : type(type), property(property)
{
}

template <typename T1, typename T2, typename T3>
Node<T1, T2, T3>::Node(const NodeT& node)
    : Inputs(node.Inputs), Outputs(node.Outputs), type(node.type), property(node.property)
{
}

template <typename T1, typename T2, typename T3>
void Node<T1, T2, T3>::SetName(std::string name)
{
    this->name = name;
}

template <typename T1, typename T2, typename T3>
int Node<T1, T2, T3>::GetInputCount() const
{
    return Inputs.size();
}

template <typename T1, typename T2, typename T3>
int Node<T1, T2, T3>::GetInputCount(const std::vector<T3>& types) const
{
    int count = 0;
    for (const EdgeT* edge : Inputs)
    {
        if (std::find(types.begin(), types.end(), edge->Property) != types.end())
        {
            count++;
        }
    }
    return count;
}

template <typename T1, typename T2, typename T3>
int Node<T1, T2, T3>::GetOutputCount() const
{
    return Outputs.size();
}

template <typename T1, typename T2, typename T3>
std::vector<typename Node<T1, T2, T3>::NodeT*> Node<T1, T2, T3>::ConnectedNodes() const
{
    std::vector<NodeT*> connectedNodes;
    for (const EdgeT* edge : Inputs)
    {
        connectedNodes.push_back(edge->From);
    }
    for (const EdgeT* edge : Outputs)
    {
        connectedNodes.push_back(edge->To);
    }
    return connectedNodes;
}

template <typename T1, typename T2, typename T3>
std::vector<typename Node<T1, T2, T3>::EdgeT*> Node<T1, T2, T3>::ConnectedEdges() const
{
    std::vector<EdgeT*> connectedEdges;
    connectedEdges.insert(connectedEdges.end(), Inputs.begin(), Inputs.end());
    connectedEdges.insert(connectedEdges.end(), Outputs.begin(), Outputs.end());
    return connectedEdges;
}

template <typename T1, typename T2, typename T3>
std::vector<typename Node<T1, T2, T3>::NodeT*> Node<T1, T2, T3>::OutputNodes() const
{
    std::vector<NodeT*> outputNodes;
    for (const EdgeT* edge : Outputs)
    {
        outputNodes.push_back(edge->To);
    }
    return outputNodes;
}

template <typename T1, typename T2, typename T3>
std::vector<typename Node<T1, T2, T3>::NodeT*> Node<T1, T2, T3>::InputNodes() const
{
    std::vector<NodeT*> inputNodes;
    for (const EdgeT* edge : Inputs)
    {
        inputNodes.push_back(edge->From);
    }
    return inputNodes;
}

template <typename T1, typename T2, typename T3>
typename Node<T1, T2, T3>::EdgeT* Node<T1, T2, T3>::GetInputEdge(T3 property, bool throwException) const
{
    for (EdgeT* edge : Inputs)
    {
        if (edge->CompareTo(property))
        {
            return edge;
        }
    }

    if (throwException)
        throw std::runtime_error("Input of the given type " + std::to_string(property) + " not found");
    else
        return nullptr;
}

template <typename T1, typename T2, typename T3>
typename Node<T1, T2, T3>::EdgeT* Node<T1, T2, T3>::GetOutputEdge(T3 property, bool throwException) const
{
    for (EdgeT* edge : Outputs)
    {
        if (edge->CompareTo(property))
        {
            return edge;
        }
    }

    if (throwException)
        throw std::runtime_error("Output of the given type " + std::to_string(property) + " not found");
    else
        return nullptr;
}

template <typename T1, typename T2, typename T3>
std::vector<typename Node<T1, T2, T3>::EdgeT*> Node<T1, T2, T3>::GetInputEdges() const
{
    return std::vector<EdgeT*>(Inputs.begin(), Inputs.end());
}

template <typename T1, typename T2, typename T3>
std::vector<typename Node<T1, T2, T3>::EdgeT*> Node<T1, T2, T3>::GetOutputEdges() const
{
    return std::vector<EdgeT*>(Outputs.begin(), Outputs.end());
}

template <typename T1, typename T2, typename T3>
typename Node<T1, T2, T3>::NodeT* Node<T1, T2, T3>::GetInput(T3 property, bool throwException) const
{
    EdgeT* edge = GetInputEdge(property, throwException);
    if (edge == nullptr)
    {
        return nullptr;
    }
    return edge->From;
}

template <typename T1, typename T2, typename T3>
typename Node<T1, T2, T3>::NodeT* Node<T1, T2, T3>::GetOutput(T3 property) const
{
    return GetOutputEdge(property)->To;
}

template <typename T1, typename T2, typename T3>
bool Node<T1, T2, T3>::CheckValidity() const
{
    for (const EdgeT* edge : Inputs)
    {
        if (edge->From->Outputs.find(edge) == edge->From->Outputs.end())
        {
            throw std::runtime_error("Input edge not found in the corresponding node");
        }
    }
    for (const EdgeT* edge : Outputs)
    {
        if (edge->To->Inputs.find(edge) == edge->To->Inputs.end())
        {
            throw std::runtime_error("Output edge not found in the corresponding node");
        }
    }
    return true;
}

template <typename T1, typename T2, typename T3>
void Node<T1, T2, T3>::AddOutput(NodeT* node, T3 property)
{
    EdgeT* newEdge = new EdgeT(this, node, property);
    Outputs.insert(newEdge);
    node->Inputs.insert(newEdge);
}

template <typename T1, typename T2, typename T3>
void Node<T1, T2, T3>::AddInput(NodeT* node, T3 property)
{
    EdgeT* newEdge = new EdgeT(node, this, property);
    Inputs.insert(newEdge);
    node->Outputs.insert(newEdge);
}

template <typename T1, typename T2, typename T3>
void Node<T1, T2, T3>::RemoveInputsExcept(NodeT* exceptNode)
{
    std::vector<EdgeT*> inputs(Inputs.begin(), Inputs.end());
    for (EdgeT* edge : inputs)
    {
        if (edge->From != exceptNode)
        {
            edge->From->Outputs.erase(edge);
            Inputs.erase(edge);
        }
    }
}

template <typename T1, typename T2, typename T3>
void Node<T1, T2, T3>::RemoveInput(NodeT* node)
{
    std::vector<EdgeT*> inputs(Inputs.begin(), Inputs.end());
    for (EdgeT* edge : inputs)
    {
        if (edge->From == node)
        {
            edge->From->Outputs.erase(edge);
            Inputs.erase(edge);
        }
    }
}