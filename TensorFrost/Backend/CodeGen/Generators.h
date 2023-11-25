#pragma once

#include "Tensor/Tensor.h"

namespace TensorFrost {
using NodeNames = std::unordered_map<const Node*, string>;

NodeNames GenerateNodeNames(const IR& ir);
string GetNodeName(const Node* node, NodeNames& names, bool compact = false);
string GetOperationListing(const IR&, bool compact = false);

string GenerateHLSL(const IR&);
string GenerateKernelHLSL(const IR&, const Lable*);

pair<string, vector<string>> GenerateC(const IR&);
string GenerateKernelC(const IR&, const Lable*);
}  // namespace TensorFrost