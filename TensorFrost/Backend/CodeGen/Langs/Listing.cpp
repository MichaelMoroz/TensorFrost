#include "Backend/CodeGen/Generators.h"
#include "IR/KernelGen.h"

namespace TensorFrost {
using namespace std;

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
		
		if (node->tensor_->type != TFType::None) {
			listing += DataTypeToString(node->tensor_->type) + " ";
		}

		if (node->tensor_->type != TFType::None) {
			//  the tensor name
			listing += node->var_name + " = ";
		}

		listing += node->name + "(";

		ArgumentManager& args = node->args;

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

		if (!node->tensor_->data.empty()) {
			listing += "data=[";
			for (int i = 0; i < node->tensor_->data.size(); i++) {
				if (i != 0) listing += ",";
				listing += to_string(node->tensor_->data[i]);
			}
			listing += "], ";
		}

		switch (node->memory_type_) {
			case MemoryType::Input:
				listing += "memory_type=input, ";
				break;
			case MemoryType::Output:
				listing += "memory_type=output, ";
				break;
			case MemoryType::Constant:
				listing += "memory_type=constant, ";
				break;
			default:
				break;
		}

		if (node->cost_ >= 0) {
			listing += "cost=" + to_string(node->cost_) + ", ";
		}

		if (node->HasBeenModified())
		{
			listing += "modified, ";
		}

		listing += ArgTypePrint("shape", ArgType::Shape);

		listing += ")\n";
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
