#include "Backend/CodeGen/Generators.h"
#include "Compiler/KernelGen.h"

namespace TensorFrost {
using namespace std;

string GetNodeString(const Node* node) {
	string listing = "";
	if (node->type != TFType::None) {
		listing += DataTypeToString(node->type) + " ";
	}

	if (node->type != TFType::None) {
		//  the tensor name
		listing += node->var_name + " = ";
	}

	listing += node->name + "(";

	const ArgumentManager& args = node->args;

	auto ArgTypePrint = [&](string name, ArgType type) {
		if (args.Has(type)) {
			string arr = name + "=[";
			for (int i = 0; i < args.Count(type); i++) {
				if (i != 0) arr += ",";
				arr += GetNodeName(args.Get(type, i), false);
			}
			arr += "], ";
			return arr;
		}
		return string();
	};

	listing += ArgTypePrint("memory", ArgType::Memory);
	listing += ArgTypePrint("inputs", ArgType::Input);
	listing += ArgTypePrint("indices", ArgType::Index);

	if (!node->data.empty()) {
		listing += "data=[";
		for (int i = 0; i < node->data.size(); i++) {
			if (i != 0) listing += ",";
			listing += to_string(node->data[i]);
		}
		listing += "], ";
	}

	if(node->flags.count() > 0) {
		listing += "flags={";
		auto flags = node->flags.get_data();
		for(auto flad_data : flags) {
			NodeProp flag = flad_data.first;
			int data = flad_data.second;
			listing += NodeFlagsToString(flag);
			if(data >= 0) {
				listing += "(" + to_string(data) + ")";
			}
			listing += ", ";
		}
		listing += "}, ";
	}

	if (node->cost_ >= 0) {
		listing += "cost=" + to_string(node->cost_) + ", ";
	}

	if(node->indexing_mode_ != IndexingMode::Clamp) {
		listing += "indexing_mode=" + IndexingModeToString(node->indexing_mode_) + ", ";
	}

	listing += ArgTypePrint("shape", ArgType::Shape);

	listing += ")";

	return listing;
}

string GetOperationListing(const IR& ir, bool compact, map<Node*, string> debug) {
	// first give unique names to all the tensors
	GenerateNodeNames(ir);
	//ClusterProp clusters = ir.GetClusterProperties();

	// now create the listing
	int prev_depth = 0;
	string listing;
	for (auto node = ir.begin(); !node.end(); node.next()) {
		if (compact) {
			if (node->name == "const") continue;
		}

		if (debug.contains(node.get())) {
			listing += "[DEBUG] " + debug[node.get()] + ": \n";
		}

		// indent
		int depth = node.depth() - 1;
		//add scope brackets
		if (depth < prev_depth) {
			for (int i = prev_depth - 1; i >= depth; i--) {
				for (int j = 0; j < i; j++) {
					listing += "  ";
				}
				listing += "}\n";
			}
		}
		else if (depth > prev_depth) {
			for (int i = prev_depth; i < depth; i++) {
				for (int j = 0; j < i; j++) {
					listing += "  ";
				}
				listing += "{\n";
			}
		}
		for (int i = 0; i < depth; i++) {
			listing += "  ";
		}
		prev_depth = depth;

		listing += GetNodeString(*node);
		listing += "\n";
	}

	for (int i = prev_depth - 1; i >= 0; i--) {
		for (int j = 0; j < i; j++) {
			listing += "  ";
		}
		listing += "}\n";
	}

	return listing;
}

}  // namespace TensorFrost
