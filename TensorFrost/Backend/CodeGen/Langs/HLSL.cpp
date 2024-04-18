#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

class HLSLGenerator : public CodeGenerator {
	unordered_map<string, string> name_map_ = {
	    {"off", "ubo.off"},
	    {"var", "ubo.var"},
	    {"dispatch_size", "ubo.dispatch_size"},

	};

 public:
	string TypeCast(string type_name, string input) override {
		return type_name + "(" + input + ")";
	}

	string GenerateAtomicOp(const string& op, const string& input_type_name,
	                        const string& output_type_name, const string& address,
	                        const string& input, const string& output) override
	{
		if (op == "InterlockedAdd_Prev") {
			return "0; InterlockedAdd(mem[" + address + "], " + input + ", " + output + ")";
		}
		else
		{
			return op + "(mem[" + address + "], " + input + ")";
		}
	}

	string GetName(const string& name) override {
		// Check if the function name is in the map
		if (name_map_.find(name) != name_map_.end()) {
			return name_map_[name];
		}

		// If not, return the original name
		return name;
	}
};

string GetHLSLHeader() { 
	return R"(
uint pcg(uint v)
{
	uint state = v * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

float pcgf(uint v)
{
	return float(pcg(v)) / float(0xffffffffu);
}

RWStructuredBuffer<uint> mem : register(u0);

struct UBO
{
	int off[32];
	int var[32];
	int dispatch_size;
};

cbuffer ubo : register(b1) { UBO ubo; }

)";
}

void GenerateHLSLKernel(Program* program, Kernel* kernel) {
	string final_source = GetHLSLHeader();

	vector<int> group_size = kernel->root->group_size;

	final_source += "[numthreads(" + to_string(group_size[0]);
	if (group_size.size() > 1) {
		final_source += ", " + to_string(group_size[1]);
	} else {
		final_source += ", 1";
	}
	if (group_size.size() > 2) {
		final_source += ", " + to_string(group_size[2]);
	} else {
		final_source += ", 1";
	}
	final_source += ")]";

	final_source += R"(

void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
  int block_id = gid.x;
  int block_thread_id0 = gtid.x;
  int block_thread_id1 = gtid.y;
  int block_thread_id2 = gtid.z;

)";

	HLSLGenerator generator;
	generator.GenerateKernelCode(kernel);
	string kernel_code = generator.AssembleString();

	final_source += AddIndent(kernel_code, "  ");

	final_source += "}\n";

    kernel->generated_code_ = final_source;
}

}  // namespace TensorFrost
