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

#include "Utility/Utility.h"

namespace TensorFrost {

using namespace std;

class Node;


class Node {
public:
    Node *parent, *child, *prev, *next;
    bool placeholder;

    Node(Node* prev = nullptr, Node* parent = nullptr) : child(nullptr), next(nullptr), placeholder(true), parent(parent), prev(prev) {}

    //initialize and create next/child placeholders
    void Initialize() {
        if(valid()) {
            throw runtime_error("Node already initialized");
        }
        if(!child) child = new Node(nullptr, this);
        if(!next) next = new Node(prev, parent);
        placeholder = false;
    }

    bool valid() {
        return !placeholder;
    }
};


class NodeIterator {
private:
    stack<Node*> parents;

public:
    Node* currentNode;

    NodeIterator(Node* node) : currentNode(node) {}

    Node* operator*() const {
        return currentNode;
    }

    //first child, then next
    NodeIterator& operator++() {
        if(!currentNode->valid()) {
            return *this;
        }

        if (currentNode->child->valid()) { //has child, go down
            parents.push(currentNode);
            currentNode = currentNode->child;
            return *this;
        }
        
        if (!currentNode->next->valid()) { //no next, try going up
            while (!parents.empty()) {
                currentNode = parents.top();
                parents.pop();
                if (currentNode->next->valid()) break;
            }
        }

        currentNode = currentNode->next;
        return *this;
    }

    bool end() {
        return !currentNode->valid();
    }

    Node* operator->() const {
        return currentNode;
    }
};

class Graph {
private:
    Node* root;

public:
    NodeIterator cursor;

    Graph() {
        root = new Node();
        root->Initialize();
        cursor = NodeIterator(root->child);
    }

    ~Graph() {
        vector<Node*> to_delete;
        for (NodeIterator it = begin(); !it.end(); ++it) {
            to_delete.push_back(*it);
        }
        for (Node* node : to_delete) {
            delete node;
        }
    }

    NodeIterator begin() {
        return NodeIterator(root->child);
    }

    void addNode() {
        if (cursor->valid()) { //already initialized, add new node before cursor
            Node* newNode = new Node(cursor->prev, cursor->parent);
            cursor->prev->next = newNode;
            cursor->prev = newNode;
            newNode->Initialize();
        } else {
            cursor->Initialize();
            cursor++;
        }
    }

    void deleteNode(Node* node) {
        if (node->valid()) {
            //if direct child of its parent
            if (node->parent->child == node) {
                node->parent->child = node->next;
            } else {
                node->prev->next = node->next;
            }

            node->next->prev = node->prev;
            
            delete node;
        }
    }

    void setCursor(Node* node) {
        cursor = NodeIterator(node);
    }

    void executeExpressionAfter(Node* node, const function<void()>&& expression) {
        NodeIterator oldCursor = cursor;
        setCursor(node->next);
        expression();
        cursor = oldCursor;
    }

    void executeExpressionBefore(Node* node, const function<void()>&& expression) {
        NodeIterator oldCursor = cursor;
        setCursor(node);
        expression();
        cursor = oldCursor;
    }

    void moveNodeTo(Node* node, Node* new_prev)
    {
        if (node->valid()) {
            if (node->parent->child == node) {
                node->parent->child = node->next;
            } else {
                node->prev->next = node->next;
            }

            node->next->prev = node->prev;

            node->prev = new_prev;
            node->next = new_prev->next;
            new_prev->next->prev = node;
            new_prev->next = node;
        }
    }
};

}  // namespace TensorFrost