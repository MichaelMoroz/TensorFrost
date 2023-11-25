#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

NodeNames GenerateNodeNames(const IR& ir) {
	NodeNames names = NodeNames();
	map<Lable*, int> cluster_var_index = map<Lable*, int>();
	int mem_index = 0;
	int cluster_index = 0;
	Lable* curent_cluster = nullptr;
	for (auto node = ir.begin(); !node.is_end(); ++node) {
		if (node->cluster_head_ != curent_cluster) {
			cluster_index++;
		}
		if (node->name == "memory") {
			names[*node] = "mem" + to_string(mem_index);
			mem_index++;
		} else {
			Lable* cluster_id = node->cluster_head_;
			int var_index = cluster_var_index[cluster_id];
			names[*node] =
			    "var" + to_string(cluster_index) + "_" + to_string(var_index);
			cluster_var_index[cluster_id]++;
		}
		curent_cluster = node->cluster_head_;
	}

	return names;
}

string GetNodeName(const Node* node, NodeNames& names, bool compact) {
	if (compact) {
		if (node->name == "const") {
			return node->tensor_->GetConstantString();
		}
	}
	return names[node];
}

inline string Tensor::GetConstantString() const {
	if (node->name == "const" || node->name == "dim_id") {
		switch (type) {
			case DataType::Float:
				return to_string(AsFloat(data[0]));
			case DataType::Int:
				return to_string(AsInt(data[0]));
			case DataType::Uint:
				return to_string(data[0]);
			default:
				return "";
		}
	} else {
		return "";
	}
}

}  // namespace TensorFrost