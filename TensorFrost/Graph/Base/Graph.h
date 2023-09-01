#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <unordered_set>
#include <vector>
#include <stdexcept>

template <typename T1, typename T2, typename T3>
class Edge;
template <typename T1, typename T2, typename T3>
class Node;
template <typename T1, typename T2, typename T3>
class Graph;

template <typename T1, typename T2, typename T3>
class Edge
{
public:
    using NodeT = Node<T1, T2, T3>;
    using EdgeT = Edge<T1, T2, T3>;

    T3 Property;
    NodeT* From;
    NodeT* To;

    Edge(NodeT* from, NodeT* to, T3 property);
    bool CompareTo(T3 y);
};

template <typename T1, typename T2, typename T3>
class Node
{
public:
    using EdgeT = Edge<T1, T2, T3>;
    using NodeT = Node<T1, T2, T3>;

    T1 type;
    T2 property;
    std::string name;

    std::unordered_set<EdgeT*> Inputs; //children
    std::unordered_set<EdgeT*> Outputs; //parents

    Node(T1 type, T2 property);
    Node(const NodeT& node);
    void SetName(std::string name);
    
    EdgeT* GetInputEdge(T3 property, bool throwException = true) const;
    EdgeT* GetOutputEdge(T3 property, bool throwException = true) const;
    NodeT* GetInput(T3 property, bool throwException = true) const;
    NodeT* GetOutput(T3 property) const;

    std::vector<NodeT*> ConnectedNodes() const;
    std::vector<EdgeT*> ConnectedEdges() const;
    std::vector<NodeT*> OutputNodes() const;
    std::vector<NodeT*> InputNodes() const;
    std::vector<EdgeT*> GetInputEdges() const;
    std::vector<EdgeT*> GetOutputEdges() const;

    int GetInputCount() const;
    int GetInputCount(const std::vector<T3>& types) const;
    int GetOutputCount() const;
    
    void AddOutput(NodeT* node, T3 property);
    void AddInput(NodeT* node, T3 property);
    void RemoveInputsExcept(NodeT* exceptNode);
    void RemoveInput(NodeT* node);

    bool CheckValidity() const;
};

template <typename T1, typename T2, typename T3>
class Cluster
{
public:
    using EdgeT = Edge<T1, T2, T3>;
    using NodeT = Node<T1, T2, T3>;
    using GraphT = Graph<T1, T2, T3>;

    std::unordered_set<NodeT*> Nodes;
};

template <typename Node, typename T2, typename T3>
class Graph
{
private:
    using EdgeT = Edge<T1, T2, T3>;
    using NodeT = Node<T1, T2, T3>;
    using GraphT = Graph<T1, T2, T3>;
    using ClusterT = Cluster<T1, T2, T3>;

    std::vector<NodeT*> Nodes;
    std::vector<EdgeT*> Edges;
    std::vector<ClusterT*> Clusters;
      
    //adding nodes and edges invalidates the topological sort
    bool IsSorted = false;
    void TopologicalSort();

public: 
    Graph();

    NodeT* AddNode(T1 type, T2 property);
    NodeT* InsetNodeIntoEdge(NodeT* node, EdgeT* edge);
    NodeT* AddEdge(NodeT* from, NodeT* to, T3 property);
    void RemoveNode(NodeT* node);
    void RemoveEdge(EdgeT* edge);

    bool AreInSameCluster(NodeT* node1, NodeT* node2) const;
    bool IsEdgeInCluster(EdgeT* edge) const;

    //return the list of nodes in topological order
    std::vector<NodeT*> GetSortedNodes() const;
    //return the list of edges in topological order
    std::vector<EdgeT*> GetSortedEdges() const;


    std::vector<NodeT*> GetOutputNodes(NodeT* node) const;
    std::vector<NodeT*> GetInputNodes(NodeT* node) const;

};

#include "Edge.tpp"
#include "Node.tpp"
#include "Graph.tpp"

#endif // GRAPH_HPP