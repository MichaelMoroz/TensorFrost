template <typename T1, typename T2, typename T3>
Edge<T1, T2, T3>::Edge(Node<T1, T2, T3>* from, Node<T1, T2, T3>* to, T3 property)
{
    this->From = from;
    this->To = to;
    this->property = property;
}

template <typename T1, typename T2, typename T3>
bool Edge<T1, T2, T3>::CompareTo(T3 y)
{
    return property == y;
}