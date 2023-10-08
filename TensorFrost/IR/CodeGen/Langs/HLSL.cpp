#pragma once

#include "IR/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

//string TranslateOperation(const Tensor* node) {
//    if (node->name == "input_memory") {
//        return "float " + GetNodeName(node) + " = 0.0;"; // Assuming float for simplicity
//    } else if (node->name == "add") {
//        return GetNodeName(node) + " = " + GetInputName(node, 0) + " + " + GetInputName(node, 1) + ";";
//    } else if (node->name == "sub") {
//        return GetNodeName(node) + " = " + GetInputName(node, 0) + " - " + GetInputName(node, 1) + ";";
//    } else if (node->name == "div") {
//        return GetNodeName(node) + " = " + GetInputName(node, 0) + " / " + GetInputName(node, 1) + ";";
//    } else if (node->name == "load") {
//        // Handling loading from memory as accessing array for simplicity
//        return GetNodeName(node) + " = " + GetInputName(node, 0) + "[" + GetIndexName(node, 0) + "][" + GetIndexName(node, 1) + "];";
//    }
//    // ... Handle other operations
//    
//    return ""; // Default for unsupported operations
//}
//
//string GetNodeName(const Tensor* tensor) {
//    // Assuming a mechanism like the one provided to get the tensor name
//    return "t" + to_string(tensor->id); // Assuming each tensor has a unique id
//}
//
//string GetInputName(const Tensor* node, int index) {
//    // Assuming a mechanism to get the input tensor based on index
//    return GetNodeName(node->GetInput(index));
//}
//
//string GetIndexName(const Tensor* node, int index) {
//    // Assuming a mechanism to get the index tensor based on index
//    return GetNodeName(node->GetIndex(index));
//}

string GenerateHLSL(const IR& ir) {
    list<Tensor*> nodes = ir.GetNodes();

    string hlslCode;
    
    // Begin HLSL function
    hlslCode += "void main() {\n";
    
    // Translate each operation into HLSL
    for (const Tensor* node : nodes) {
        hlslCode += node->name + "\n";
    }
    
    // End HLSL function
    hlslCode += "}\n";
    
    return hlslCode;
}

}